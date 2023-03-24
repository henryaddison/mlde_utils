import matplotlib
import matplotlib.pyplot as plt
import metpy.plots.ctables
import seaborn as sns

from . import cp_model_rotated_pole

# precip_clevs = [0, 1, 2.5, 5, 7.5, 10, 15, 20, 30, 40,
#      50, 70, 100, 150, 200, 250, 300, 400, 500, 600, 750, 1000]
# precip_norm, precip_cmap = metpy.plots.ctables.registry.get_with_boundaries('precipitation', precip_clevs)
precip_clevs = [0, 0.1, 1, 2.5, 5, 7.5, 10, 15, 20, 30, 40, 50, 70, 100, 150, 200]
precip_cmap = matplotlib.colors.ListedColormap(
    metpy.plots.ctables.colortables["precipitation"][: len(precip_clevs) - 1],
    "precipitation",
)
precip_norm = matplotlib.colors.BoundaryNorm(precip_clevs, precip_cmap.N)

STYLES = {
    "precip": {"cmap": precip_cmap, "norm": precip_norm},
    "logBlues": {"cmap": "Blues", "norm": matplotlib.colors.LogNorm()},
}


def create_map_fig(grid_spec, width=None, height=None):
    if width is None:
        width = len(grid_spec[0]) * 5.5
    if height is None:
        height = len(grid_spec) * 5.5
    subplot_kw = dict(projection=cp_model_rotated_pole)
    return plt.subplot_mosaic(
        grid_spec,
        figsize=(width, height),
        subplot_kw=subplot_kw,
        constrained_layout=True,
    )


def plot_map(da, ax, title="", style="logBlues", add_colorbar=False, **kwargs):
    if style is not None:
        kwargs = STYLES[style] | kwargs
    pcm = da.plot.pcolormesh(ax=ax, add_colorbar=add_colorbar, **kwargs)
    ax.set_title(title)
    ax.coastlines()
    # ax.gridlines(draw_labels={"bottom": "x", "left": "y"}, x_inline=False, y_inline=False)#, xlabel_style=dict(fontsize=24), ylabel_style=dict(fontsize=24))
    return pcm


def freq_density_plot(ax, ds, target_pr, grouping_key="model", diagnostics=False):
    pred_pr = ds["pred_pr"]

    hrange = (
        min(pred_pr.min().values, target_pr.min().values),
        max(pred_pr.max().values, target_pr.max().values),
    )
    _, bins, _ = target_pr.plot.hist(
        ax=ax,
        bins=50,
        density=True,
        color="black",
        alpha=0.2,
        label="CPM",
        log=True,
        range=hrange,
    )
    for group_value in pred_pr[grouping_key].values:
        pred_pr.sel({grouping_key: group_value}).plot.hist(
            ax=ax,
            bins=bins,
            density=True,
            alpha=0.75,
            histtype="step",
            label=f"{grouping_key} {group_value}",
            log=True,
            range=hrange,
            linewidth=2,
            linestyle="-",
        )

    ax.set_title("Log density of samples and CPM precip")
    ax.set_xlabel("Precip (mm day-1)")
    ax.tick_params(axis="both", which="major")
    if diagnostics:
        text = f"""
        # Timestamps: {pred_pr["time"].count().values}
        # Samples: {pred_pr.count().values}
        # Targets: {target_pr.count().values}
        % Samples == 0: {(((pred_pr == 0).sum()/pred_pr.count()).values*100).round()}
        % Targets == 0: {(((target_pr == 0).sum()/target_pr.count()).values*100).round()}
        % Samples < 1e-5: {(((pred_pr < 1e-5).sum()/pred_pr.count()).values*100).round()}
        % Targets < 1e-5: {(((target_pr < 1e-5).sum()/target_pr.count()).values*100).round()}
        % Samples < 0.1: {(((pred_pr < 0.1).sum()/pred_pr.count()).values*100).round()}
        % Targets < 0.1: {(((target_pr < 0.1).sum()/target_pr.count()).values*100).round()}
        % Samples < 1: {(((pred_pr < 1).sum()/pred_pr.count()).values*100).round()}
        % Targets < 1: {(((target_pr < 1).sum()/target_pr.count()).values*100).round()}
        Sample max: {pred_pr.max().values.round()}
        Target max: {target_pr.max().values.round()}
        """
        ax.text(0.7, 0.5, text, fontsize=8, transform=ax.transAxes)
    ax.legend()
    # ax.set_aspect(aspect=1)


def qq_plot(
    ax,
    target_quantiles,
    sample_quantiles,
    grouping_key="model",
    title="Sample vs CPM quantiles",
    xlabel="CPM precip (mm day-1)",
    ylabel="Sample precip (mm day-1)",
    tr=200,
    bl=0,
    guide_label="Ideal",
    **scatter_args,
):
    # if guide_label is not None:
    ax.plot(
        [bl, tr],
        [bl, tr],
        color="black",
        linestyle="--",
        label=guide_label,
        alpha=0.2,
    )

    # ax.set_xlim(bl, tr)

    for label, group_quantiles in sample_quantiles.groupby(grouping_key):
        ax.scatter(
            target_quantiles,
            group_quantiles.mean(dim="sample_id"),
            **(dict(label=label, alpha=0.75, marker="x") | scatter_args),
        )
        data = (
            group_quantiles.squeeze()
            .to_pandas()
            .reset_index()
            .melt(
                id_vars="quantile", value_vars=list(group_quantiles["sample_id"].values)
            )
            .merge(target_quantiles.to_pandas().rename("cpm_quantile").reset_index())
        )
        sns.lineplot(data=data, x="cpm_quantile", y="value", errorbar="sd", ax=ax)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.set_aspect(aspect=1)
