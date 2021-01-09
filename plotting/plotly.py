import seaborn as sns
from plotly import graph_objects as go, figure_factory as ff


class SeabornColorPalette:
    palette_names = [
        "viridis",
        "plasma",
        "inferno",
        "magma",

        "Greys",
        "Purples",
        "Blues",
        "Greens",
        "Oranges",
        "Reds",
        "YlOrBr",
        "YlOrRd",
        "OrRd",
        "PuRd",
        "RdPu",
        "BuPu",
        "GnBu",
        "PuBu",
        "YlGnBu",
        "PuBuGn",
        "BuGn",
        "YlGn",

        "binary",
        "gist_yarg",
        "gist_gray",
        "gray",
        "bone",
        "pink",
        "spring",
        "summer",
        "autumn",
        "winter",
        "cool",
        "Wistia",
        "hot",
        "afmhot",
        "gist_heat",
        "copper",

        "PiYG",
        "PRGn",
        "BrBG",
        "PuOr",
        "RdGy",
        "RdBu",
        "RdYlBu",
        "RdYlGn",
        "Spectral",
        "coolwarm",
        "bwr",
        "seismic",

        "Pastel1",
        "Pastel2",
        "Paired",
        "Accent",
        "Dark2",
        "Set1",
        "Set2",
        "Set3",
        "tab10",
        "tab20",
        "tab20b",
        "tab20c",

        "flag",
        "prism",
        "ocean",
        "gist_earth",
        "terrain",
        "gist_stern",
        "gnuplot",
        "gnuplot2",
        "CMRmap",
        "cubehelix",
        "brg",
        "hsv",
        "gist_rainbow",
        "rainbow",
        # "jet",
        "nipy_spectral",
        "gist_ncar"
    ]


    @classmethod
    def to_plotly_rgb(cls, colorpalette, num_color):
        palette = sns.color_palette(colorpalette, num_color)
        rgb = ['rgb({},{},{})'.format(*[int(x * 256) for x in rgb])
               for rgb in palette]
        return rgb


def seaborn_color_palette_2_plotly_rgb(colorpalette, n_colors):
    palette = sns.color_palette(colorpalette, n_colors)
    rgb = ['rgb({},{},{})'.format(*[int(x*256) for x in rgb])
           for rgb in palette]
    return rgb


def plotly_boxplot(data, x, y, pallete='Pastel1', title=None, width=1200, height=700, showmeans=False, dark=False):
    x_list = data[x].unique()
    color = seaborn_color_palette_2_plotly_rgb(colorpalette=pallete, n_colors=len(x_list))

    if dark:
        template = "plotly_dark"
    else:
        template = "plotly"

    fig = go.Figure()
    for i, x_ in enumerate(x_list):
        df = data[data[x] == x_]
        fig.add_trace(
            go.Box(y=df[y], name=x_, marker_color=color[i],
                   boxmean=showmeans)
        )

    if title is not None:
        fig.update_layout(
            title=title,
            xaxis_title=x,
            yaxis_title=y,
            width=width,
            height=height,
            template=template
        )

    # fig.show()
    return fig


def plotly_heatmap(data, cmap='Blues', side=500, dark=False):
    if dark:
        template = "plotly_dark"
    else:
        template = "plotly"

    fig = ff.create_annotated_heatmap(z=data.values[::-1], x=list(data.columns), y=list(data.index)[::-1],
                                      hoverinfo='none', colorscale=cmap)

    fig.update_layout(width=side, height=side,
                      xaxis_title="Cor.", yaxis_title="Pre.",
                      template=template)
    # fig.show()
    return fig


def get_trace_per_model(data, model, color):
    data_ = data[data["Model"] == model]

    trace = go.Box(
        x=data_["Label"],
        y=data_["F-measure"],
        name=model,
        marker_color=color,
        boxmean=True
    )

    return trace


def plotly_multi_boxplot(data, x, y, hue, pallet='Pastel1', title=None, width=1200, height=700, showmeans=False, dark=False):
    hue_list = data[hue].unique()

    trace_list = []

    color = seaborn_color_palette_2_plotly_rgb(pallet, len(hue_list))

    for index, hue_ in enumerate(hue_list):
        data_ = data[data[hue] == hue_]
        trace = go.Box(
            x=data_[x],
            y=data_[y],
            name=hue_,
            marker_color=color[index],
            boxmean=showmeans
        )

        trace_list += [trace]

    layout = go.Layout(boxmode='group')
    fig = go.Figure(data=trace_list, layout=layout)

    if title is not None:
        fig.update_layout(title=title,
                          xaxis_title=x,
                          yaxis_title=y,
                          width=width,
                          height=height)

    return fig