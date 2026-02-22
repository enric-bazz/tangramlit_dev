import numpy as np
from typing import Tuple, List, Iterable, Union

def split_train_val_test_genes(
    adata_sc,
    adata_st,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    random_state: int | None = None,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split shared genes into train / val / test sets.

    Procedure:
    1) intersect sc and st genes (case-insensitive)
    2) split into (train+val) and test
    3) split (train+val) into train and val

    Ratios are with respect to the full shared set.

    Args:
        adata_sc: AnnData (single-cell)
        adata_st: AnnData (spatial)
        train_ratio: fraction of genes used for training
        val_ratio: fraction of genes used for validation
        random_state: RNG seed

    Returns:
        train_genes, val_genes, test_genes
    """
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Require: train_ratio > 0, val_ratio >= 0, train_ratio + val_ratio < 1")

    rng = np.random.default_rng(random_state)

    # case-insensitive mapping, preserve original names
    sc_genes = {g.lower(): g for g in adata_sc.var.index}
    st_genes = {g.lower(): g for g in adata_st.var.index}

    shared_keys = sorted(set(sc_genes) & set(st_genes))
    if len(shared_keys) == 0:
        raise ValueError("No shared genes between adata_sc and adata_st")

    genes = np.array([sc_genes[k] for k in shared_keys])

    n_total = len(genes)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    perm = rng.permutation(n_total)

    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]

    train_genes = genes[train_idx].tolist()
    val_genes = genes[val_idx].tolist()
    test_genes = genes[test_idx].tolist()

    return train_genes, val_genes, test_genes



def kfold_gene_splits(
    genes: List[str],
    k: Union[int, str] = 10,
    random_state: int | None = None,
) -> Iterable[Tuple[List[str], List[str]]]:
    """
    Generate k-fold or leave-one-out (LOO) gene splits.

    Args:
        genes: list of gene names
        k: number of folds (int >=2) or "loo" for leave-one-out CV
        random_state: RNG seed for reproducibility

    Yields:
        train_genes, test_genes
    """
    genes = np.array(genes)
    n = len(genes)

    if isinstance(k, str) and k.lower() == "loo":
        for i in range(n):
            test_idx = [i]
            train_idx = [j for j in range(n) if j != i]
            yield genes[train_idx].tolist(), genes[test_idx].tolist()
        return

    if not isinstance(k, int) or k < 2 or k > n:
        raise ValueError(f"k must be an int >=2 and <= number of genes ({n}), or 'loo'")

    rng = np.random.default_rng(random_state)
    perm = rng.permutation(n)
    folds = np.array_split(perm, k)

    for i in range(k):
        test_idx = folds[i]
        train_idx = np.concatenate(folds[:i] + folds[i+1:])
        yield genes[train_idx].tolist(), genes[test_idx].tolist()

