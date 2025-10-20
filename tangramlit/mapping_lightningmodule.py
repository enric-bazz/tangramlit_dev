import random
import numpy as np
import sklearn
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn.functional import softmax, cosine_similarity
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.callbacks import ProgressBar
from tqdm import tqdm

from . import validation_metrics as vm
from . import utils as ut

"""
Lightning module for Tangram
"""

class MapperLightning(pl.LightningModule):
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
            lambda_ct_islands: Optional. Strength of ct islands enforcement. Default is 0.
            lambda_ct_islands: Optional. Strength of ct islands enforcement. Default is 0.
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
        self.d_prior = None  # density prior
        self.getis_ord_G_star_ref = None  # LISA
        self.moran_I_ref = None
        self.gearys_C_ref = None
        self.epoch_values = None  # epoch loss terms

        # Define density criterion
        self._density_criterion = nn.KLDivLoss(reduction="sum")

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
            ct_encode = batch['A']
            self.register_buffer("ct_encode", ct_encode, persistent=False)

            # Set random seed if specified
            if self.hparams.random_state is not None:
                torch.manual_seed(self.hparams.random_state)
                random.seed(self.hparams.random_state)
                np.random.seed(self.hparams.random_state)

            # Initialize mapping matrix M (nn.Parameter.data)
            self.M = nn.Parameter(torch.randn(n_cells, n_spots))

            # Initialize filter F
            if self.hparams.filter:
                self.F = nn.Parameter(torch.randn(n_cells))
                # To test uniform init: nn.Parameter(torch.ones(n_cells) / n_cells)

            # Define uniform density prior
            self.d_prior = torch.ones(n_spots) / n_spots

            # Spatial weights
            # Mapping: (condition, buffer name, kwargs for compute_spatial_weights)
            buffer_map = [
                (self.hparams.lambda_neighborhood_g1 > 0, "voxel_weights", dict(standardized=True, self_inclusion=True)),
                (self.hparams.lambda_ct_islands > 0, "neighborhood_filter", dict(standardized=False, self_inclusion=False)),
                (self.hparams.lambda_moran > 0 or self.hparams.lambda_geary > 0, "spatial_weights_morangeary", dict(standardized=True, self_inclusion=False)),
                (self.hparams.lambda_getis_ord > 0, "spatial_weights_getisord", dict(standardized=False, self_inclusion=True)),
            ]
            # Compute and register buffers (not loaded in state dict)
            for cond, name, kwargs in buffer_map:
                if cond:
                    tensor = self._compute_spatial_weights(graph_conn, graph_dist, **kwargs)
                    self.register_buffer(name, tensor, persistent=False)

            # Compute LISA on ground truth (reference values)
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
            Output tensors are computed on the Dataloader batch and inherit its device, thus should move to device with data.

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
            m2 = torch.sum((G - G.mean(dim=0)) ** 2, dim=0) / (n_spots - 1)
            weighted_diff_sq = self.spatial_weights_morangeary.unsqueeze(2) * ((G[None, :, :] - G[:, None, :]) ** 2)
            # NOTE: None entries add a singleton dimension, Torch broadcasts to (n_spots, n_spots, n_cells)
            gearys_C = weighted_diff_sq.sum(dim=(0, 1)) / (2 * m2)

        return getis_ord_G_star, moran_I, gearys_C
    

    def forward(self):
        """
            Compute the mapping probabilities.

            Returns softmax probabilities and, if in filter mode, the filtered probabilities and filter values.

            NOTE: The original Tangram algorithm uses the filtered M matrix only to compute the estimated density.
            All the other terms (cossim, kl regularizer, etc.) are computed on the non-filtered softmax matrix.
        """
        if self.hparams.filter:
            F_probs = torch.sigmoid(self.F)
            M_probs = softmax(self.M, dim=1)

            return M_probs, M_probs * F_probs[:, None], F_probs  # broadcasting is more efficient than torch.diag(F_probs) @ M_probs
        else:
            return softmax(self.M, dim=1)


    def training_step(self, batch):
        """
            Training step using data from the datamodule.
            NOTE: Refinement loss terms are compute only if their lambda > 0, else term = 0.
            NOTE: As of now, the main loss cannot be turned off.

            Returns:
                step_output (dict): Dictionary containing the loss terms and other values to be logged.
            Defines:
                self.epoch_values (dict): Dictionary containing the loss terms.
                    All optional terms are discarded if their respective lambda_ parameter is set to 0.
                    Compulsory terms are always included.
        """
        # Get data
        S_train = batch['S']  # single-cell data
        G_train = batch['G']  # spatial data

        # Forward step
        if self.hparams.filter:
            M_probs, M_probs_filtered, F_probs = self()  # Get softmax probabilities and filter probabilities
        else:
            M_probs = self()  # Get softmax probabilities

        # Loss computation
        # Expression term
        if self.hparams.filter:
            G_pred = M_probs_filtered.T @ S_train
        else:
            G_pred = M_probs.T @ S_train
        gv_term = self.hparams.lambda_g1 * cosine_similarity(G_pred, G_train, dim=0).mean()
        vg_term = self.hparams.lambda_g2 * cosine_similarity(G_pred, G_train, dim=1).mean()
        expression_term = gv_term + vg_term

        # Density term
        if self.hparams.filter:
            d_pred = torch.log(M_probs_filtered.sum(axis=0) / (F_probs.sum()))
        else:
            d_pred = torch.log(M_probs.sum(axis=0) / self.M.shape[0])
        density_term = self.hparams.lambda_d * self._density_criterion(d_pred, self.d_prior)

        # Entropy regularizer term
        regularizer_term = self.hparams.lambda_r * (torch.log(M_probs) * M_probs).sum()

        # l1 and l2 regularization terms
        l1_term = self.hparams.lambda_l1 * self.M.abs().sum()
        l2_term = self.hparams.lambda_l2 * (self.M ** 2).sum()
        # NOTE: These terms act on the alignment matrix M, not the softmax matrix M_probs

        # Sparsity weighted expression term
        if self.hparams.lambda_sparsity_g1 > 0:
            mask = G_train != 0
            gene_sparsity = mask.sum(axis=0) / G_train.shape[0]
            gene_sparsity = 1 - gene_sparsity.reshape((-1,))
            gv_sparsity_term = self.hparams.lambda_sparsity_g1 * (
                        (cosine_similarity(G_pred, G_train, dim=0) * (1 - gene_sparsity)) / (1 - gene_sparsity).sum()).sum()
        else:
            gv_sparsity_term = 0

        # Spatial neighborhood-based gene expression term
        if self.hparams.lambda_neighborhood_g1 > 0:
            gv_neighborhood_term = self.hparams.lambda_neighborhood_g1 * cosine_similarity(self.voxel_weights @ G_pred,
                                                                                    self.voxel_weights @ G_train,
                                                                                    dim=0).mean()
        else:
            gv_neighborhood_term = 0

        # Spatial autocorrelation statistics
        getis_ord_G_star_pred, moran_I_pred, gearys_C_pred = self._spatial_local_indicators(G_pred)

        # Spatial autcorrelation terms
        getis_ord_term, moran_term, gearys_term = 0, 0, 0
        if self.hparams.lambda_getis_ord > 0:
            getis_ord_term = self.hparams.lambda_getis_ord * cosine_similarity(self.getis_ord_G_star_ref,
                                                                        getis_ord_G_star_pred, dim=0).mean()
        if self.hparams.lambda_moran > 0:
            moran_term = self.hparams.lambda_moran * cosine_similarity(self.moran_I_ref, moran_I_pred, dim=0).mean()
        if self.hparams.lambda_geary > 0:
            gearys_term = self.hparams.lambda_geary * cosine_similarity(self.gearys_C_ref, gearys_C_pred, dim=0).mean()

        # Cell type island enforcement
        if self.hparams.lambda_ct_islands > 0:
            ct_map = (M_probs.T @ self.ct_encode)
            ct_island_term = self.hparams.lambda_ct_islands * (torch.max((ct_map) - (self.neighborhood_filter @ ct_map),
                                                                    torch.tensor([0], dtype=torch.float32)).mean())
        else:
            ct_island_term = 0

        # Total loss
        total_loss = (
            - expression_term
            + density_term
            - regularizer_term
            + l1_term
            + l2_term
            - gv_sparsity_term
            - gv_neighborhood_term
            - getis_ord_term
            - moran_term
            - gearys_term
            + ct_island_term
        )

        # Filter terms
        if self.hparams.filter:
            # Count term: abs( sum(f_i, over cells i) - n_target_cells)
            count_term = self.hparams.lambda_count * torch.abs(F_probs.sum() - self.hparams.target_count)
            # Filter regularizer: sum(f_i - f_i^2, over cells i)
            f_reg = self.hparams.lambda_f_reg * (F_probs - F_probs * F_probs).sum()
            # Update total loss
            total_loss += count_term + f_reg 

        # Create the dictionary of loss terms
        step_output = {
            "loss": total_loss,
            "main_loss": gv_term,
            "vg_reg": vg_term if self.hparams.lambda_g2 > 0 else None,
            "kl_reg": density_term if self.hparams.lambda_d > 0 else None,
            "entropy_reg": regularizer_term if self.hparams.lambda_r > 0 else None,
            "l1_term": l1_term if self.hparams.lambda_l1 > 0 else None,
            "l2_term": l2_term if self.hparams.lambda_l2 > 0 else None,
            "sparsity_term": gv_sparsity_term if self.hparams.lambda_sparsity_g1 > 0 else None,
            "neighborhood_term": gv_neighborhood_term if self.hparams.lambda_neighborhood_g1 > 0 else None,
            "getis_ord_term": getis_ord_term if self.hparams.lambda_getis_ord > 0 else None,
            "moran_term": moran_term if self.hparams.lambda_moran > 0 else None,
            "geary_term": gearys_term if self.hparams.lambda_geary > 0 else None,
            "ct_island_term": ct_island_term if self.hparams.lambda_ct_islands > 0 else None,
        }
        # Filter terms
        if self.hparams.filter:
            filter_terms = {
                "count_reg": count_term if self.hparams.lambda_count > 0 else None,
                "filt_reg": f_reg if self.hparams.lambda_f_reg > 0 else None,
            }
            step_output.update(filter_terms)
            # Filter history values
            self.filter_history['filter_values'].append(F_probs.detach().cpu().numpy())
            self.filter_history['n_cells'].append((F_probs > 0.5).sum().item())

        # Create a state-persistent dictionary with only non-None values
        self.epoch_values = {k: v.detach().cpu().item() for k, v in step_output.items() if v is not None}

        return step_output

    def on_train_epoch_end(self, N=100):
        """
            Handle training history tracking, logging and displaying.

            Args:
                N (int): number of epochs between each print. Default: 100.
        """
        # Append to training history dictionary
        for key in self.loss_history.keys():
            if key in self.epoch_values:
                self.loss_history[key].append(self.epoch_values[key])

        # Log to tensorboard logger
        self.log_dict(self.epoch_values, prog_bar=False, logger=True, on_epoch=True)

        # Print every N epochs
        if self.current_epoch % N == 0:
            losses = {
                k: v.item() if torch.is_tensor(v) else v
                for k, v in self.trainer.logged_metrics.items()  # logged_metrics: metrics logged via log() with the logger argument set
            }
            print(f"Epoch {self.current_epoch}: {losses}")

        # Track learning rate scheduling
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", lr, prog_bar=False)


    def configure_optimizers(self):
        """
            Set optimizer and learning rate scheduler.
        """
        # Set optimizer on parameters M and F (if in filter mode) or only on M (if in refined mode)
        if self.hparams.filter:
            optimizer = torch.optim.Adam([self.M, self.F], lr=self.hparams.learning_rate)
        else:
            optimizer = torch.optim.Adam([self.M], lr=self.hparams.learning_rate)
        # Set the learning rate scheduler on main loss monitoring
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10),
            'monitor': 'loss',  # monitor total loss for smoother trajectory
            'interval': 'epoch',
            'frequency': 1,
        }
        return [optimizer], [scheduler]

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
           Produces two different behaviors depending on the trainer function.
           The common behaviour is to compute the validation terms in val_dict: score, spatial sparsity weighted score, auc, entropy.
           - if trainer.fit() is called (fn == 'fit'): log val_dict into 'train' logger for callbacks monitoring
           - if trainer.validate() is called (fn == 'validate'): 
                * computes additional validation metrics in metrics_dict: ssim, pcc, rmse, js
                * updates val_dict and logs it into 'val' logger

           Args:
              batch: full validation batch.

           Returns:
               Logs a dictionary of validation terms or validation metrics depending on the function.
               If trainer.state.fn == 'fit' --> Validation Score, Sparsity-weighted Score, AUC, Entropy
               If trainer.state.fn == 'validate' --> val_SSIM, val_PCC, val_RMSE, val_JS
       """
        # Get validation data
        S_val = batch['S']  # single-cell data (test genes)
        G_val = batch['G']  # spatial data (test genes)

        # Run "forward" pass and prediction
        M_probs = softmax(self.M, dim=1)  # mapping matrix (learned on training genes)
        G_pred = torch.matmul(M_probs.t(), S_val)  # projected single-cell data (test genes)

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
        auc_score, _ = vm.poly2_auc(gv_scores, gene_sparsity, plot_auc=plot_auc)  # skip coordinates
        # Entropy of the mapping probabilities
        prob_entropy = - torch.mean(torch.sum((torch.log(M_probs) * M_probs), dim=1) / np.log(M_probs.shape[1]))
        # Validation dictionary
        val_dict = {'val_score': gv_scores.mean(),
                    'val_sparsity-weighted_score': sp_sparsity_weighted_scores.mean(),
                    'val_AUC': auc_score,
                    'val_entropy': prob_entropy}

        if self.trainer.state.fn == 'fit':  # if trainer.fit() is called
            # Log on progress bar and train logger
            self.log_dict(val_dict, prog_bar=True, logger=True, on_epoch=True)

        if self.trainer.state.fn == 'validate':  # if trainer.validate() is called
            # Define imputed and raw spatial expression arrays
            imputed_data = G_pred.detach().cpu().numpy()
            raw_data = G_val.detach().cpu().numpy()
            # Compute validation metrics (all np arrays averaged into floats)
            metrics_dict = {'val_SSIM': vm.ssim(raw_data, imputed_data).mean(),
                            'val_PCC': vm.pearsonr(raw_data, imputed_data).mean(),
                            'val_RMSE': vm.RMSE(raw_data, imputed_data).mean(),
                            'val_JS': vm.JS(raw_data, imputed_data).mean()}
            # Add to validation dictionary
            val_dict.update(metrics_dict)
            # Log on validation logger
            self.log_dict(val_dict, prog_bar=False, logger=True, on_epoch=True)

    def on_validation_epoch_end(self):
        """
            Print validation terms at the end of each validation epoch during training.
        """
        if self.trainer.state.fn == 'fit':
            losses = {
                k: v.item() if torch.is_tensor(v) else v
                for k, v in self.trainer.progress_bar_metrics.items()
            }
            print(f"\nValidation {self.current_epoch}: {losses}")

    def on_validation_start(self):
        """
            Print dataset dimensions during validation sanity check.
        """
        if self.trainer.sanity_checking:  # first validation_step() call
            batch = next(iter(self.trainer.datamodule.val_dataloader()))
            print(f"\nValidating with {batch['genes_number']} genes")
            print(f"S matrix shape: {batch['S'].shape}")
            print(f"G matrix shape: {batch['G'].shape}")

# TODO: return both aggregated and gene-specific metrics during training and validation
# TODO: work with on_load_checkpoint, on_save_checkpoin methods to handle checkpoints


"""
Overwrite progress bar class to get a single persistent bar over epochs.
"""
class EpochProgressBar(ProgressBar):
    def __init__(self):
        super().__init__()
        self.bar = None

    def on_train_start(self, trainer, pl_module):
        self.bar = tqdm(total=trainer.max_epochs, desc="Trainig", leave=True, dynamic_ncols=True)

    def on_train_epoch_end(self, trainer, pl_module):
        self.bar.update(1)

    def on_train_end(self, trainer, pl_module):
        self.bar.close()
