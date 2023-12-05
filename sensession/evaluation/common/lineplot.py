import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def lineplot(
    data: pl.DataFrame,
    x_label: str,
    x_col: str,
    y_label: str,
    y_col: str,
    ax: plt.Axes,
    title: str,
    hue_col: str = None,
    leg_bbox=(1.02, 0.00),
    leg_loc="lower left",
    labels=None,  # automatically deduced if not given
    lower_ylim=None,
    marker="",
):
    img = sns.lineplot(
        x=x_col,
        y=y_col,
        hue=hue_col,
        data=data.to_pandas(),
        ax=ax,
        legend="full",
        marker=marker,
    )

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.draw()

    # Set ticks
    ax.tick_params(axis="x", labelrotation=60)
    for label in ax.get_xticklabels()[::2]:
        label.set_visible(False)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: int(x)))

    # Set title and labels
    if hue_col:
        if labels is not None:
            img.legend(loc=leg_loc, bbox_to_anchor=leg_bbox, ncol=1, labels=labels)
        else:
            img.legend(loc=leg_loc, bbox_to_anchor=leg_bbox, ncol=1)

    if lower_ylim:
        ax.set_ylim(bottom=lower_ylim)

    img.set_title(title)
    ax.set_xlabel(x_label, fontsize=15)
    ax.set_ylabel(y_label, fontsize=15)

    return img
