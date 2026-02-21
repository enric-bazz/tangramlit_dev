import random
import numpy as np
import sklearn
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import auc
import logging
import lightning.pytorch as lp
from lightning.pytorch import LightningModule
import torch
import torch.nn as nn
from torch.nn.functional import softmax, cosine_similarity
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lightning.pytorch.callbacks import ProgressBar
from tqdm import tqdm

from .loss import TangramLoss

"""
Lightning module for Tangram
"""

### LightningModule class ###

class MapperLightning(LightningModule):
    def __init__(
            self,
            filter=False,
            learning_rate=0.1,
            num_epochs=1000,
            random_state=None,
            lambda_d=0,
            lambda_g1=1,
            lambda_g2=0,
            lambda_r=0,
            lambda_l1=0,
            lambda_l2=0,
            lambda_count=1,
            lambda_f_reg=1,
            target_count=None,
            lambda_sparsity_g1=0,
            lambda_neighborhood_g1=0,
            lambda_getis_ord=0,
            lambda_geary=0,
            lambda_moran=0,
            lambda_ct_islands=0,
    ):
        """
        Lightning Module initializer.

        Args:
            filter (bool): Whether or not the cell filter is trained. Default is False.
            random_state (int): Optional. pass an int to reproduce training. Default is None.
            learning_rate (float): Optional. Learning rate for the optimizer. Default is 0.1.
            num_epochs (int): Optional. Number of epochs. Default is 1000.
            lambda_d (float): Optional. Hyperparameter for the density term of the optimizer. Default is 0.
            lambda_g1 (float): Optional. Hyperparameter for the gene-voxel similarity term of the optimizer. Default is 1.
            lambda_g2 (float): Optional. Hyperparameter for the voxel-gene similarity term of the optimizer. Default is 0.
            lambda_r (float): Optional. Strength of entropy regularizer. An higher entropy promotes probabilities of each cell peaked over a narrow portion of space. lambda_r = 0 corresponds to no entropy regularizer. Default is 0.
            lambda_l1 (float): Optional. Strength of L1 regularizer. Default is 0.
            lambda_l2 (float): Optional. Strength of L2 regularizer. Default is 0.
            lambda_count (float): Optional. Regularizer for the count term. Default is 1. Only valid when mode == 'constrained'
            lambda_f_reg (float): Optional. Regularizer for the filter, which promotes Boolean values (0s and 1s) in the filter. Only valid when mode == 'constrained'. Default is 1.
            target_count (int): Optional. The number of cells to be filtered. Default is None.
            lambda_sparsity_g1 (float): Optional. Strength of sparsity weighted gene expression comparison. Default is 0.
            lambda_neighborhood_g1 (float): Optional. Strength of neighborhood weighted gene expression comparison. Default is 0.
            lambda_getis_ord (float): Optional. Strength of Getis-Ord G* preservation. Default is 0.
            lambda_geary (float): Optional. Strength of Geary's C preservation. Default is 0.
            lambda_moran (float): Optional. Strength of Moran's I preservation. Default is 0.
            lambda_ct_islands (float): Optional. Strength of ct islands enforcement. Default is 0.
        """

        super().__init__()
        # Pass all args as self.hparams attributes
        self.save_hyperparameters()

        # Allocate fields for setup() attributes
        self.M = None
        # self.M = nn.Parameter(torch.empty(0), requires_grad=True)  # to pre-allocate parameter
        if self.hparams.filter:
            self.F = None
            # self.F = nn.Parameter(torch.empty(0), requires_grad=True)  # to pre-allocate parameter
        # self.d_prior = None  # density prior
        self.getis_ord_G_star_ref = None  # LISA
        self.moran_I_ref = None
        self.gearys_C_ref = None
        self.epoch_values = None  # epoch loss terms

        # Instantiate the loss class
        self.loss_fn = TangramLoss(self.hparams)

        # Create the training history dictionary
        self.loss_history = {
            'loss': [],
            'main_loss': [],
            'vg_reg': [],
            'kl_reg': [],
            'entropy_reg': [],
            'l1_term': [],
            'l2_term': [],
            'sparsity_term': [],
            'neighborhood_term': [],
            'getis_ord_term': [],
            'moran_term': [],
            'geary_term': [],
            'ct_island_term': [],
        }
        # Add filter terms and filter history tracking if applicable
        if self.hparams.filter:
            self.loss_history.update({
                'count_reg': [],
                'filt_reg': [],
            })
            self.filter_history = {
                'filter_values': [],  # filter values per epoch
                'n_cells': []  # number of cells that pass the filter per epoch
            }

        # Create validation history dictionary
        self.val_history = {
            'val_score': [],
            'val_sparsity-weighted_score': [],
            'val_AUC': [],
            'val_entropy': [],
        }

    def setup(self, stage=None):
        """
            Initialize mapping matrices using data dimensions from datamodule.
            Define uniform density prior.
            Define weighting tensors for refined mode.
            Called after datamodule is set but before training starts.
        """
        if stage == 'fit' or stage is None:
            # Get a train batch from the datamodule to determine dimensions
            dataloader_train = self.trainer.datamodule.train_dataloader()
            batch = next(iter(dataloader_train))
            S, G = batch['S'], batch['G']
            n_cells, n_genes_sc = S.shape
            n_spots, n_genes_st = G.shape

            # Get spatial graph from batch
            graph_conn, graph_dist = batch['spatial_graph_conn'], batch['spatial_graph_dist']

            # Get single cell annotation OHE and register buffer (no state dict)
            if self.hparams.lambda_ct_islands > 0:
                ct_encode = batch['A']
                self.register_buffer("ct_encode", ct_encode, persistent=False)

            # Set random seed if specified
            if self.hparams.random_state is not None:
                torch.manual_seed(self.hparams.random_state)
                random.seed(self.hparams.random_state)
                np.random.seed(self.hparams.random_state)
                # For reproducibility on GPU
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(self.hparams.random_state)
                    torch.cuda.manual_seed_all(self.hparams.random_state)

            # Initialize mapping matrix M
            self.M = nn.Parameter(torch.randn(n_cells, n_spots))

            # Initialize filter F
            if self.hparams.filter:
                self.F = nn.Parameter(torch.randn(n_cells))
                # To test uniform init: nn.Parameter(torch.ones(n_cells) / n_cells)

            # Define uniform density prior (register as buffer so it moves to correct device)
            self.register_buffer("d_prior", torch.ones(n_spots) / n_spots, persistent=False)

            # Spatial weights
            # Mapping: (condition, buffer name, kwargs for compute_spatial_weights)
            buffer_map = [
                (self.hparams.lambda_neighborhood_g1 > 0, "voxel_weights", dict(standardized=True, self_inclusion=True)),
                (self.hparams.lambda_ct_islands > 0, "neighborhood_filter", dict(standardized=False, self_inclusion=False)),
                (self.hparams.lambda_moran > 0 or self.hparams.lambda_geary > 0, "spatial_weights_morangeary", dict(standardized=True, self_inclusion=False)),
                (self.hparams.lambda_getis_ord > 0, "spatial_weights_getisord", dict(standardized=False, self_inclusion=True)),
            ]

            for cond, name, kwargs in buffer_map:
                if cond:
                    tensor = self._compute_spatial_weights(graph_conn, graph_dist, **kwargs)
                    self.register_buffer(name, tensor, persistent=False)
                    size_mib = tensor.element_size() * tensor.nelement() / (1024 ** 2)
                    #print(f"Registered buffer '{name}' | shape={tuple(tensor.shape)} | size={size_mib:.2f} MiB")

            # LISA on ground truth (reference values)
            self.getis_ord_G_star_ref, self.moran_I_ref, self.gearys_C_ref = self._spatial_local_indicators(G)
   

    def _compute_spatial_weights(self, connectivities, distances, standardized, self_inclusion):
        """
        Compute spatial weights matrix.
        Contains either row-standardized distances or binary adjacencies. Can include self or not (1 or 0 diagonal).

        Args:
            connectivities (torch.tensor) = spatial graph connectivities (adjacency values of 0/1)
            distances (torch.tensor) = spatial graph distances (euclidean)
            standardized (bool) = Whether the spatial distances matrix is row-standardized or not.
            self_inclusion (bool) = Whether the spatial weights matrix includes the diagonal (1s).

        Returns:
            A (n_spots, n_spots) torch.tensor with:
                - adjacency-filtered row-normalised spatial distances if standardize == True
                - connectivity weights (0/1) if standardize == False
            with diagonal entries based on the value of self_inclusion.

        """
        if standardized:
            # Row-normalize distances
            distances_norm = torch.from_numpy(
                sklearn.preprocessing.normalize(distances, norm="l1", axis=1, copy=False).astype("float32")
                )
            # Mask normalized distances with connectivity (keep only neighbor links)
            spatial_weights = connectivities.multiply(distances_norm)
        else:
            spatial_weights = connectivities
        if self_inclusion:
            spatial_weights += np.eye(spatial_weights.shape[0], dtype=np.float32)

        return spatial_weights
    

    def _spatial_local_indicators(self, G):
        """
        Compute spatial local indicators for the given spatial gene expression matrix.
        Defined as a method to access lambda_getis_ord, lambda_moran, lambda_geary, and spatial_weights attributes.
        G is call-dependent so is passed as an argument.
        Output tensors are computed on the same device as G.

        Args:
            G (torch.tensor), shape = (number_spots, number_genes): Spatial gene expression matrix defined in the DataModule.

        Returns:
            A tuple containing local indicators, each a torch.tensor with shape (n_genes, n_spots).
        """
        # Getis Ord G*
        getis_ord_G_star = None
        if self.hparams.lambda_getis_ord > 0:
            getis_ord_G_star = (self.spatial_weights_getisord @ G) / G.sum(dim=0)

        # Moran's I
        moran_I = None
        if self.hparams.lambda_moran > 0:
            z = G - G.mean(dim=0)
            moran_I = (G.shape[0] * z * (self.spatial_weights_morangeary @ z)) / torch.sum(z * z, dim=0)

        # Geary's C
        gearys_C = None
        if self.hparams.lambda_geary > 0:
            n_spots = G.shape[0]
            W = self.spatial_weights_morangeary  # (n_spots, n_spots), dense manageable
            m2 = torch.sum((G - G.mean(dim=0)) ** 2, dim=0) / (n_spots - 1)
            WG = W @ G                            # (n_spots, n_genes)
            term1 = (W.sum(dim=1, keepdim=True) * (G ** 2)).sum(dim=0)  # sum_i w_i. * G_i^2
            term2 = (G * WG).sum(dim=0)                               # sum_i G_i * (sum_j w_ij G_j)
            numerator = term1 - term2
            gearys_C = numerator / m2

        return getis_ord_G_star, moran_I, gearys_C
    

    def forward(self):
        """
        Compute softmax mapping probabilities and, if in filter mode, the filtered probabilities and filter values.
        """
        if self.hparams.filter:
            F_probs = torch.sigmoid(self.F)
            M_probs = softmax(self.M, dim=1)

            return M_probs, M_probs * F_probs[:, None], F_probs  # broadcasting is more efficient than torch.diag(F_probs) @ M_probs
        else:
            return softmax(self.M, dim=1)


    def training_step(self, batch):
        """
        Training step using data from the PairedDataModule class.
        Args:
            batch: full training batch.
        Returns:
            loss_dict (dict): Dictionary containing loss terms (total loss in 'loss').
        Defines:
            self.epoch_values (dict): State-persistent dictionary containing loss terms.
                Terms are discarded if their respective lambda_ parameter is set to 0.
        """
        S_train = batch['S']
        G_train = batch['G']

        if self.hparams.filter:
            M_probs, M_probs_filtered, F_probs = self()
            G_pred = M_probs_filtered.T @ S_train

            # track filter history
            self.filter_history['filter_values'].append(F_probs.detach().cpu().numpy())
            self.filter_history['n_cells'].append((F_probs > 0.5).sum().item())
        else:
            M_probs = self()
            G_pred = M_probs.T @ S_train

        if (self.hparams.lambda_getis_ord + self.hparams.lambda_moran + self.hparams.lambda_geary > 0):
            getis_ord_G_star_pred, moran_I_pred, gearys_C_pred = self._spatial_local_indicators(G_pred)

        loss_dict = self.loss_fn(
            G=G_train,
            G_pred=G_pred,
            M=self.M,
            M_probs=M_probs,
            M_probs_filtered=M_probs_filtered if self.hparams.filter else None,
            F_probs=F_probs if self.hparams.filter else None,
            d_prior=getattr(self, "d_prior", None),
            voxel_weights=getattr(self, "voxel_weights", None),
            ct_encode=getattr(self, "ct_encode", None),
            neighborhood_filter=getattr(self, "neighborhood_filter", None),
            getis_ord_G_star_ref=getattr(self, "getis_ord_G_star_ref", None),
            moran_I_ref=getattr(self, "moran_I_ref", None),
            gearys_C_ref=getattr(self, "gearys_C_ref", None),
            getis_ord_G_star_pred=getis_ord_G_star_pred,
            moran_I_pred=moran_I_pred,
            gearys_C_pred=gearys_C_pred,
        )
       
        self.epoch_values = {}
        for k, v in loss_dict.items():
            if v is None:
                continue
            if torch.is_tensor(v):
                self.epoch_values[k] = v.detach().cpu().item()
            else:
                # try to cast numeric-like values to float, otherwise keep as-is
                try:
                    self.epoch_values[k] = float(v)
                except Exception:
                    self.epoch_values[k] = v

        return loss_dict

    def on_train_epoch_end(self, N=100, verbose=False):
        """
        Handle training history tracking, logging and displaying.

        Args:
            N (int): number of epochs between each print. Default: 100.
            verbose (bool): whether to print epoch losses or not. Default: False.
        """
        # Append to training history dictionary
        for key in self.loss_history.keys():
            if key in self.epoch_values:
                self.loss_history[key].append(self.epoch_values[key])

        # Log to tensorboard logger
        self.log_dict(self.epoch_values, prog_bar=False, logger=True, on_epoch=True)

        # Print every N epochs
        if verbose and (self.current_epoch % N == 0):
            losses = {
                k: v.item() if torch.is_tensor(v) else v
                for k, v in self.trainer.logged_metrics.items()  # logged_metrics: metrics logged via log() with the logger argument set
            }
            print(f"Epoch {self.current_epoch}: {losses}")




    def configure_optimizers(self):
        """
            Set optimizer and learning rate scheduler.
        """
        if self.hparams.filter:
            optimizer = torch.optim.Adam([self.M, self.F], lr=self.hparams.learning_rate)
        else:
            optimizer = torch.optim.Adam([self.M], lr=self.hparams.learning_rate)

        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10),
            'monitor': 'loss',  # monitor total loss for smoother trajectory
            'interval': 'epoch',
            'frequency': 1,
        }
        return [optimizer], [scheduler]

    @property
    def device(self):
        """
        Get the device the module is on (handles CPU/GPU transparently).
        """
        return self.M.device

    def on_train_start(self):
        """
        Print dataset dimensions at the start of training.
        """
        batch = next(iter(self.trainer.train_dataloader))
        print(f"\nTraining with {batch['genes_number']} genes")
        print(f"S matrix shape: {batch['S'].shape}")
        print(f"G matrix shape: {batch['G'].shape}")


    def validation_step(self, batch):
        """
        Compute the validation terms in val_dict: score, spatial sparsity weighted score, auc, entropy.

        Args:
            batch: full validation batch.

        Returns:
            Logs a dictionary of validation terms:  Validation Score, Sparsity-weighted Score, AUC, Entropy
       """
        # Get validation data
        S_val = batch['S']  # single-cell data (test genes)
        G_val = batch['G']  # spatial data (test genes)

        # Run "forward" pass and prediction
        if self.hparams.filter:
            M_probs, M_probs_filtered, _ = self() 
            G_pred = M_probs_filtered.T @ S_val
        else:
            M_probs = self() 
            G_pred = M_probs.T @ S_val


        # Genes scores
        gv_scores = cosine_similarity(G_pred, G_val, dim=0)
        # Sparsity level of genes
        mask = G_val != 0
        gene_sparsity = mask.sum(axis=0) / G_val.shape[0]
        gene_sparsity = 1 - gene_sparsity.reshape((-1,))
        # Sparsity weighted score
        sp_sparsity_weighted_scores = ((gv_scores * (1 - gene_sparsity)) / (1 - gene_sparsity).sum())
        # AUC (plot conditioned on trainer function)
        plot_auc = self.trainer.state.fn == 'validate'
        auc_score, _ = poly2_auc(gv_scores, gene_sparsity, plot_auc=plot_auc)  # skip coordinates
        # Entropy of the mapping probabilities
        prob_entropy = - torch.mean(torch.sum((torch.log(M_probs) * M_probs), dim=1) / np.log(M_probs.shape[1]))
        # Validation dictionary
        val_dict = {'val_score': gv_scores.mean(),
                    'val_sparsity-weighted_score': sp_sparsity_weighted_scores.sum(),
                    'val_AUC': auc_score,
                    'val_entropy': prob_entropy,
                    }

        self.log_dict(val_dict, prog_bar=True, logger=True, on_epoch=True)

        return val_dict

    def on_validation_start(self):
        """
        Print dataset dimensions during validation sanity check, that is the first validation_step() call.
        """
        if self.trainer.sanity_checking:
            batch = next(iter(self.trainer.datamodule.val_dataloader()))
            print(f"\nValidating with {batch['genes_number']} genes")
            print(f"S matrix shape: {batch['S'].shape}")
            print(f"G matrix shape: {batch['G'].shape}")

    def on_validation_epoch_end(self, verbose=False):
        """
        Print validation terms at the end of each validation epoch during training.
        """
        if self.trainer.state.fn == 'fit':
            losses = {
                k: v.item() if torch.is_tensor(v) else v
                for k, v in self.trainer.progress_bar_metrics.items()
            }
            if verbose:
                print(f"\nValidation {self.current_epoch}: {losses}")
            for key, value in losses.items():
                if key not in self.val_history:
                    self.val_history[key] = []
                if not self.trainer.sanity_checking:
                    self.val_history[key].append(value)


# TODO: return both aggregated and gene-specific metrics during training and validation
# TODO: work with on_load_checkpoint, on_save_checkpoin methods to handle checkpoints


### ProgressBar class ###

class EpochProgressBar(ProgressBar):
    """
    Overwrite progress bar class to get a single persistent bar over epochs.
    """
    def __init__(self):
        super().__init__()
        self.bar = None

    def on_train_start(self, trainer, pl_module):
        self.bar = tqdm(total=trainer.max_epochs, desc="Trainig", leave=True, dynamic_ncols=True)

    def on_train_epoch_end(self, trainer, pl_module):
        self.bar.update(1)

    def on_train_end(self, trainer, pl_module):
        self.bar.close()


### Utility functions ###

def poly2_auc(gv_scores, gene_sparsity, pol_deg=2, plot_auc=False):
    """
    Compute Tangram most important evaluation metric.
    Fit a 2nd-degree polynomial between gv_scores and gene_sparsity,
    clip to [0,1], and return the area under the curve (AUC).

    Args:
        gv_scores : array-like or torch.Tensor
            Cosine similarity scores per gene.
        gene_sparsity : array-like or torch.Tensor
            Gene sparsity values per gene (0-1).
        pol_deg : int
            Degree of polynomial (default 2).
        plot_auc (bool): Wether to plot the polynomial fit in the (sparsity, score) plane or not. Default is False.

    Returns:
        auc_score (float): Area under the polynomial curve over x in [0,1].
        auc_coordinates (tuple): AUC fitted coordinates and raw coordinates (test_score vs. sparsity_st coordinates)
        plot: polyfit curve and genes (sparsity, score) scatter plot
    """

    # Convert to numpy arrays
    xs = np.array(gv_scores).flatten()
    ys = np.array(gene_sparsity).flatten()

    # Fit polynomial
    pol_cs = np.polyfit(xs, ys, pol_deg)
    pol = np.poly1d(pol_cs)

    # Sample polynomial on [0,1]
    pol_xs = np.linspace(0, 1, 50)
    pol_ys = pol(pol_xs)

    # Clip values to [0,1]
    pol_ys = np.clip(pol_ys, 0, 1)

    # Include real roots where y=0 inside [0,1]
    roots = pol.r
    for r in roots:
        if np.isreal(r) and 0 <= r.real <= 1:
            pol_xs = np.append(pol_xs, r.real)
            pol_ys = np.append(pol_ys, 0)

    # Sort x values for proper integration
    sort_idx = np.argsort(pol_xs)
    pol_xs = pol_xs[sort_idx]
    pol_ys = pol_ys[sort_idx]

    # Compute AUC
    auc_score = auc(pol_xs, pol_ys)

    # Coordinates
    auc_coordinates = ((pol_xs, pol_ys), (xs, ys))

    if plot_auc:
        plt.figure(figsize=(6, 5))

        plt.plot(pol_xs, pol_ys, c='r')
        sns.scatterplot(x=xs, y=ys, alpha=0.5, edgecolors='face')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.gca().set_aspect(.5)
        plt.xlabel('score')
        plt.ylabel('spatial sparsity')
        plt.tick_params(axis='both', labelsize=8)
        plt.title('Prediction on validation transcriptome')

        textstr = 'auc_score={}'.format(np.round(auc_score, 3))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
        # place a text box in upper left in axes coords
        plt.text(0.03, 0.1, textstr, fontsize=11, verticalalignment='top', bbox=props)
        plt.show()

    return float(auc_score), auc_coordinates