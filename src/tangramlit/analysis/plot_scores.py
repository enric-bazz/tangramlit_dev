import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ..mapping.lit_mapper import poly2_auc

def plot_training_scores(df_g, bins=10, alpha=0.7):
    """
        Plots the 4-panel training diagnosis plot. Restricted on genes flagged as 'is_training'.

        Args:
            df_g (pandas.DataFrame): Contains overlap genes sparsity/score values produced by mapping_utils.compare_spatial_gene_expr()
            bins (int or string): Optional. Default is 10.
            alpha (float): Optional. Ranges from 0-1, and controls the opacity. Default is 0.7.

        Returns:
            None
    """
    df_g = df_g.loc[df_g['is_training']]

    fig, axs = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
    axs_f = axs.flatten()

    # set limits for axis
    axs_f[0].set_ylim([0.0, 1.0])
    for i in range(1, len(axs_f)):
        axs_f[i].set_xlim([0.0, 1.0])
        axs_f[i].set_ylim([0.0, 1.0])

    axs_f[0].set_title('Training scores for single genes')
    sns.histplot(data=df_g, y="score", bins=bins, ax=axs_f[0], color="coral")

    axs_f[1].set_title("score vs sparsity (single cells)")
    sns.scatterplot(
        data=df_g,
        y="score",
        x="sparsity_sc",
        ax=axs_f[1],
        alpha=alpha,
        color="coral",
    )

    axs_f[2].set_title("score vs sparsity (spatial)")
    sns.scatterplot(
        data=df_g,
        y="score",
        x="sparsity_st",
        ax=axs_f[2],
        alpha=alpha,
        color="coral",
    )

    axs_f[3].set_title("score vs sparsity (sp - sc)")
    sns.scatterplot(
        data=df_g,
        y="score",
        x="sparsity_diff",
        ax=axs_f[3],
        alpha=alpha,
        color="coral",
    )

    plt.tight_layout()
    plt.show()


def plot_auc_curve(df_g, plot_train=True, plot_validation=True):
    """
        Plots auc curve of non-training genes score. Test genes are either input or deduced from df_g['is_training'].

        Args:
            df_g (pandas.DataFrame): returned by compare_spatial_gene_expr(adata_ge, adata_sp).
            plot_train (bool): if True, include genes flagged as training (df_g['is_training'] == True).
            plot_test (bool): if True, include genes flagged as NOT training (df_g['is_training'] == False). Default True.

        Returns:
            None
        """
    # Use all rows in df_g and split by the boolean flags
    df_sel = df_g.copy()

    # At least one of the plotting flags must be True
    if not plot_train and not plot_validation:
        raise ValueError('At least one of plot_train or plot_validation must be True.')

    def prepare_group(df, mask_col):
        grp = df.loc[df[mask_col] == True].copy()
        # remove zero-score entries (poly fit / auc routine requires non-zero scores)
        grp = grp.loc[grp['score'] != 0]
        return grp

    # Create separate figures for train and validation groups.
    auc_train = np.nan
    auc_val = np.nan

    fig_train = None
    fig_val = None

    produced_any = False

    # Training group
    if plot_train and 'is_training' in df_sel.columns:
        grp_train = prepare_group(df_sel, 'is_training')
        if not grp_train.empty:
            auc_train, ((pol_x_tr, pol_y_tr), (xs_tr, ys_tr)) = poly2_auc(grp_train['score'], grp_train['sparsity_st'], plot_auc=False)
            fig_train = plt.figure(figsize=(6, 5))
            plt.plot(pol_x_tr, pol_y_tr, c='r', label=f'poly fit')
            sns.scatterplot(x=xs_tr, y=ys_tr, alpha=0.6, edgecolors='face', color='blue', label='genes')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0])
            plt.gca().set_aspect(.5)
            plt.xlabel('score')
            plt.ylabel('spatial sparsity')
            plt.tick_params(axis='both', labelsize=8)
            plt.legend()
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
            plt.text(0.03, 0.1, f'AUC={np.round(auc_train,3)}', fontsize=11, verticalalignment='top', bbox=props)
            plt.title('Training quadratic polynomial fit')
            produced_any = True

    # Validation group
    if plot_validation and 'is_validation' in df_sel.columns:
        grp_val = prepare_group(df_sel, 'is_validation')
        if not grp_val.empty:
            auc_val, ((pol_x_val, pol_y_val), (xs_val, ys_val)) = poly2_auc(grp_val['score'], grp_val['sparsity_st'], plot_auc=False)
            fig_val = plt.figure(figsize=(6, 5))
            plt.plot(pol_x_val, pol_y_val, c='r', label=f'poly fit')
            sns.scatterplot(x=xs_val, y=ys_val, alpha=0.6, edgecolors='face', color='blue', label='genes')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0])
            plt.gca().set_aspect(.5)
            plt.xlabel('score')
            plt.ylabel('spatial sparsity')
            plt.tick_params(axis='both', labelsize=8)
            plt.legend()
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
            plt.text(0.03, 0.1, f'AUC={np.round(auc_val,3)}', fontsize=11, verticalalignment='top', bbox=props)
            plt.title('Validation quadratic polynomial fit')
            produced_any = True

    if not produced_any:
        raise ValueError('No genes with non-null score in the requested groups, CosSim not computable.')

    # Return the two separate figures (either may be None)
    return (fig_train, fig_val), (auc_train, auc_val)


def plot_score_SA_corr(df_g, plot_train=True, plot_validation=True, plot_fit=True):
    """
    Plots score vs spatial-autocorrelation statistics (Moran's I and Geary's C) and
    computes linear correlation coefficients for training/validation gene groups.

    Args:
        df_g (pandas.DataFrame): DataFrame returned by compare_spatial_gene_expr(), must contain
            columns 'score' and one or both of 'moranI' and 'gearyC' (or 'geary').
        plot_train (bool): If True, produce plots for genes flagged as training ('is_training').
        plot_validation (bool): If True, produce plots for genes flagged as validation ('is_validation').
        plot_fit (bool): If True, overlay a linear fit line on the scatter plots.

    Returns:
        tuple: (figures_tuple, correlations_tuple)
            figures_tuple = (fig_train_moran, fig_val_moran, fig_train_geary, fig_val_geary)
            correlations_tuple = (corr_train_moran, corr_val_moran, corr_train_geary, corr_val_geary)
    """
    df_sel = df_g.copy()

    if not plot_train and not plot_validation:
        raise ValueError('At least one of plot_train or plot_validation must be True.')

    # Determine available spatial-autocorr columns
    moran_col = 'moranI' if 'moranI' in df_sel.columns else None
    geary_col = 'gearyC' if 'gearyC' in df_sel.columns else ('geary' if 'geary' in df_sel.columns else None)

    if moran_col is None and geary_col is None:
        raise ValueError("Neither 'moranI' nor 'gearyC'/'geary' found in dataframe.")

    def prepare_group(df, mask_col, stat_col):
        grp = df.loc[df[mask_col] == True].copy()
        grp = grp.loc[grp['score'] != 0]
        if stat_col is not None:
            grp = grp.loc[~grp[stat_col].isnull()]
        return grp

    def safe_corr(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        mask = (~np.isnan(x)) & (~np.isnan(y))
        if mask.sum() < 2:
            return np.nan
        try:
            return float(np.corrcoef(x[mask], y[mask])[0, 1])
        except Exception:
            return np.nan

    # placeholders for figures and correlations
    fig_train_moran = None
    fig_val_moran = None
    fig_train_geary = None
    fig_val_geary = None

    corr_train_moran = np.nan
    corr_val_moran = np.nan
    corr_train_geary = np.nan
    corr_val_geary = np.nan

    produced_any = False

    # Training group
    if plot_train and 'is_training' in df_sel.columns:
        if moran_col is not None:
            grp = prepare_group(df_sel, 'is_training', moran_col)
            if not grp.empty:
                xs = grp['score']
                ys = grp[moran_col]
                corr_train_moran = safe_corr(xs, ys)
                fig_train_moran = plt.figure(figsize=(6, 5))
                sns.scatterplot(x=ys, y=xs, alpha=0.6, edgecolors='face', color='blue', label='genes')
                if plot_fit and xs.size > 1:
                    mask = (~np.isnan(xs)) & (~np.isnan(ys))
                    if mask.sum() > 1 and np.unique(ys[mask]).size > 1:
                        coeffs = np.polyfit(ys[mask], xs[mask], 1)
                        xmin = float(np.min(ys[mask]))
                        xmax = float(np.max(ys[mask]))
                        rng = xmax - xmin
                        delta = rng * 0.05 if rng > 0 else max(0.1, abs(xmin) * 0.05)
                        fit_x = np.linspace(xmin - delta, xmax + delta, 200)
                        fit_y = np.polyval(coeffs, fit_x)
                        plt.plot(fit_x, fit_y, c='r', label='linear fit')
                plt.ylim([0.0, 1.0])
                plt.gca().set_aspect(.5)
                plt.xlabel(moran_col)
                plt.ylabel('score')
                plt.tick_params(axis='both', labelsize=8)
                plt.legend()
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
                plt.text(0.03, 0.1, f'corr={np.round(corr_train_moran,3)}', fontsize=11, verticalalignment='top', bbox=props)
                plt.title('Training Moran I vs score')
                produced_any = True

        if geary_col is not None:
            grp = prepare_group(df_sel, 'is_training', geary_col)
            if not grp.empty:
                xs = grp['score']
                ys = grp[geary_col]
                corr_train_geary = safe_corr(xs, ys)
                fig_train_geary = plt.figure(figsize=(6, 5))
                sns.scatterplot(x=ys, y=xs, alpha=0.6, edgecolors='face', color='blue', label='genes')
                if plot_fit and xs.size > 1:
                    mask = (~np.isnan(xs)) & (~np.isnan(ys))
                    if mask.sum() > 1 and np.unique(ys[mask]).size > 1:
                        coeffs = np.polyfit(ys[mask], xs[mask], 1)
                        xmin = float(np.min(ys[mask]))
                        xmax = float(np.max(ys[mask]))
                        rng = xmax - xmin
                        delta = rng * 0.05 if rng > 0 else max(0.1, abs(xmin) * 0.05)
                        fit_x = np.linspace(xmin - delta, xmax + delta, 200)
                        fit_y = np.polyval(coeffs, fit_x)
                        plt.plot(fit_x, fit_y, c='r', label='linear fit')
                plt.ylim([0.0, 1.0])
                plt.gca().set_aspect(.5)
                plt.xlabel(geary_col)
                plt.ylabel('score')
                plt.tick_params(axis='both', labelsize=8)
                plt.legend()
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
                plt.text(0.03, 0.1, f'corr={np.round(corr_train_geary,3)}', fontsize=11, verticalalignment='top', bbox=props)
                plt.title('Training Geary C vs score')
                produced_any = True

    # Validation group
    if plot_validation and 'is_validation' in df_sel.columns:
        if moran_col is not None:
            grp = prepare_group(df_sel, 'is_validation', moran_col)
            if not grp.empty:
                xs = grp['score']
                ys = grp[moran_col]
                corr_val_moran = safe_corr(xs, ys)
                fig_val_moran = plt.figure(figsize=(6, 5))
                sns.scatterplot(x=ys, y=xs, alpha=0.6, edgecolors='face', color='blue', label='genes')
                if plot_fit and xs.size > 1:
                    mask = (~np.isnan(xs)) & (~np.isnan(ys))
                    if mask.sum() > 1 and np.unique(ys[mask]).size > 1:
                        coeffs = np.polyfit(ys[mask], xs[mask], 1)
                        xmin = float(np.min(ys[mask]))
                        xmax = float(np.max(ys[mask]))
                        rng = xmax - xmin
                        delta = rng * 0.05 if rng > 0 else max(0.1, abs(xmin) * 0.05)
                        fit_x = np.linspace(xmin - delta, xmax + delta, 200)
                        fit_y = np.polyval(coeffs, fit_x)
                        plt.plot(fit_x, fit_y, c='r', label='linear fit')
                plt.ylim([0.0, 1.0])
                plt.gca().set_aspect(.5)
                plt.xlabel(moran_col)
                plt.ylabel('score')
                plt.tick_params(axis='both', labelsize=8)
                plt.legend()
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
                plt.text(0.03, 0.1, f'corr={np.round(corr_val_moran,3)}', fontsize=11, verticalalignment='top', bbox=props)
                plt.title('Validation Moran I vs score')
                produced_any = True

        if geary_col is not None:
            grp = prepare_group(df_sel, 'is_validation', geary_col)
            if not grp.empty:
                xs = grp['score']
                ys = grp[geary_col]
                corr_val_geary = safe_corr(xs, ys)
                fig_val_geary = plt.figure(figsize=(6, 5))
                sns.scatterplot(x=ys, y=xs, alpha=0.6, edgecolors='face', color='blue', label='genes')
                if plot_fit and xs.size > 1:
                    mask = (~np.isnan(xs)) & (~np.isnan(ys))
                    if mask.sum() > 1 and np.unique(ys[mask]).size > 1:
                        coeffs = np.polyfit(ys[mask], xs[mask], 1)
                        xmin = float(np.min(ys[mask]))
                        xmax = float(np.max(ys[mask]))
                        rng = xmax - xmin
                        delta = rng * 0.05 if rng > 0 else max(0.1, abs(xmin) * 0.05)
                        fit_x = np.linspace(xmin - delta, xmax + delta, 200)
                        fit_y = np.polyval(coeffs, fit_x)
                        plt.plot(fit_x, fit_y, c='r', label='linear fit')
                plt.ylim([0.0, 1.0])
                plt.gca().set_aspect(.5)
                plt.xlabel(geary_col)
                plt.ylabel('score')
                plt.tick_params(axis='both', labelsize=8)
                plt.legend()
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
                plt.text(0.03, 0.1, f'corr={np.round(corr_val_geary,3)}', fontsize=11, verticalalignment='top', bbox=props)
                plt.title('Validation Geary C vs score')
                produced_any = True

    if not produced_any:
        raise ValueError('No genes with non-null score/statistic in the requested groups, cannot compute correlations.')

    figures = (fig_train_moran, fig_val_moran, fig_train_geary, fig_val_geary)
    corrs = (corr_train_moran, corr_val_moran, corr_train_geary, corr_val_geary)
    return figures, corrs


def plot_score_histograms(df_g, bins=10, alpha=0.7, plot_train=True, plot_validation=True, remove_zero=True):
    """
    Plot histograms of `score` values for training and validation gene groups.

    Args:
        df_g (pandas.DataFrame): DataFrame containing at least 'score' and optional
            boolean columns 'is_training' and 'is_validation'. If those flags are
            missing, histograms for the corresponding group will be None.
        bins (int or sequence): Number of bins or bin edges for the histogram.
        alpha (float): Opacity for histogram bars.
        plot_train (bool): If True, produce the training histogram.
        plot_validation (bool): If True, produce the validation histogram.
        remove_zero (bool): If True, drop rows with score == 0 before plotting.

    Returns:
        tuple: (fig_train, fig_val) where each is a matplotlib Figure or None.
    """
    df_sel = df_g.copy()

    if not plot_train and not plot_validation:
        raise ValueError('At least one of plot_train or plot_validation must be True.')

    fig_train = None
    fig_val = None

    # helper to prepare series
    def get_scores(df, mask_col=None):
        if mask_col is not None and mask_col in df.columns:
            s = df.loc[df[mask_col] == True, 'score'].copy()
        else:
            s = df['score'].copy()
        if remove_zero:
            s = s.loc[s != 0]
        s = s.dropna()
        return s

    # Training histogram
    if plot_train and 'is_training' in df_sel.columns:
        scores_tr = get_scores(df_sel, 'is_training')
        if not scores_tr.empty:
            fig_train = plt.figure(figsize=(6, 4))
            sns.histplot(scores_tr, bins=bins, kde=False, color='coral', alpha=alpha)
            plt.xlim([0.0, 1.0])
            plt.ylim(bottom=0)
            plt.xlabel('score')
            plt.ylabel('count')
            plt.title('Training gene score distribution')
            plt.tight_layout()

    # Validation histogram
    if plot_validation and 'is_validation' in df_sel.columns:
        scores_val = get_scores(df_sel, 'is_validation')
        if not scores_val.empty:
            fig_val = plt.figure(figsize=(6, 4))
            sns.histplot(scores_val, bins=bins, kde=False, color='steelblue', alpha=alpha)
            plt.xlim([0.0, 1.0])
            plt.ylim(bottom=0)
            plt.xlabel('score')
            plt.ylabel('count')
            plt.title('Validation gene score distribution')
            plt.tight_layout()

    return fig_train, fig_val