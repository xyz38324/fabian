import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def scatterplot(
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
    lower_ylim=None,
):
    img = sns.scatterplot(
        x=x_col,
        y=y_col,
        hue=hue_col,
        data=data.to_pandas(),
        ax=ax,
    )
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.draw()

    # Set ticks
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: int(x)))

    ax.tick_params(axis="x", labelrotation=60)
    for label in ax.get_xticklabels()[::2]:
        label.set_visible(False)

    # Set title and labels
    if hue_col:
        img.legend(loc=leg_loc, bbox_to_anchor=leg_bbox, ncol=1)

    if lower_ylim:
        ax.set_ylim(bottom=lower_ylim)

    img.set_title(title)
    ax.set_xlabel(x_label, fontsize=15)
    ax.set_ylabel(y_label, fontsize=15)
