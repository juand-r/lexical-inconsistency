"""
Code for visualizations

"""
#TODO: add code for stacked barplot to show token ratios across layers.

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
import numpy as np

def f(L1, L2, c):
    """Filter subset of L1 where entries of L2 have taxonomic==c"""
    return [i for ii, i in enumerate(L1) if L2[ii].taxonomic == c]

def plot_correlations2(L, gen, disc):
    taxonomic = [i.taxonomic for i in L]
    df = pd.DataFrame(
        {
            "generator": [i[-1].tolist() for i in gen[:]],
            "discriminator": [i[-1].tolist() for i in disc[:]],
            "taxonomic": taxonomic,
        }
    )


    plt.figure(figsize=(10, 8))

    # Adjust the color palette for better contrast
    g = sns.jointplot(
        data=df,
        x="generator",
        y="discriminator",
        hue="taxonomic",
        #palette={"yes": "blue", "no": "red"}
        #palette="coolwarm",  # or choose a different color palette
        kind="scatter", #scatter, kde or hist
        height=8,  # This controls the size of the jointplot
        alpha=0.9,
        #contour_kws={"levels": 15, "colors": ["blue", "orange"], "alpha": 0.7},
        #fill=True
        s=50,
        #marginal_kws={'bw': 2.0}
    )
    
    g.ax_joint.grid(True, linestyle='--', linewidth=0.7)  # Grid with dashed lines
    # Customize axis labels and title
    g.set_axis_labels("Generator log odds", "Validator log odds", fontsize=23)
    #g.ax_joint.legend(fontsize=14)  # Set legend font size
# Customize the legend
    handles, labels = g.ax_joint.get_legend_handles_labels()
    #custom_order = ["yes", "no"]  # Modify this list to control the order

    # Reorder handles and labels based on the custom order
    #reordered_handles = [handles[labels.index(label)] for label in custom_order]
    #reordered_labels = custom_order
    custom_labels = {"yes": "Positive", "no": "Negative"}

    # Replace the original labels with custom ones
    new_labels = [custom_labels[label] if label in custom_labels else label for label in labels]

    custom_order = ["yes", "no"]  # Modify this list to control the order

# Reorder handles and labels based on the custom order
    reordered_handles = [handles[labels.index(label)] for label in custom_order]
    reordered_labels = [custom_labels[label] if label in custom_labels else label for label in custom_order]



    # Define custom labels
    custom_labels = {"yes": "Positive", "no": "Negative"}
    new_labels = [custom_labels[label] if label in custom_labels else label for label in labels]
    #plt.tick_params(axis='both', which='both', length=7, width=2, labelsize=14)
    plt.tick_params(axis='both', which='both', labelsize=16)


    g.ax_joint.legend(
        handles=reordered_handles,
        labels=reordered_labels,
        title="Class",  # Set legend title
        fontsize=16,  # Set font size for legend labels
        title_fontsize=18,  # Set font size for legend title
        loc='upper left',  # Position the legend
        #bbox_to_anchor=(1, 1),  # Adjust position outside the plot
        markerscale=1.5,  # Scale the size of the markers in the legend
        frameon=True,  # Add a frame around the legend
        fancybox=True,  # Round corners for the legend frame
        edgecolor='black',  # Set the border color of the legend
        facecolor='lightgray',  # Set the background color of the legend
    )


    #g.fig.suptitle("Discriminator vs Generator with Taxonomic Classification", fontsize=16, weight='bold')
    #g.fig.subplots_adjust(top=0.95)

    # Optional: Adjust axis limits if necessary
    g.ax_joint.set_xlim(-35, 10.3)
    g.ax_joint.set_ylim(-7, 4.2)

    plt.show()

def plot_correlations(L, gen, disc):
    """
    Plot correlations between generator and discriminator log-odds.
    Also computes pearson correlations, overall, and for positive
    and negative classes.

    Parameters
    ----------
    L : list
        The list of hypernym items obtained via load_noun_pair_data()
    gen : list (torch.Tensor)
        List of tensors; each tensor is log-odds over layers
    disc : list (torch.Tensor)
        List of tensors; each tensor is log-odds over layers
    """
    taxonomic = [i.taxonomic for i in L]
    df = pd.DataFrame(
        {
            "generator": [i[-1].tolist() for i in gen[:]],
            "discriminator": [i[-1].tolist() for i in disc[:]],
            "taxonomic": taxonomic,
        }
    )

    g = sns.jointplot(
        data=df,
        x="generator",
        y="discriminator",
        hue="taxonomic",
        palette={"yes": "blue", "no": "orange"},
        kind="kde",
        # xlim=(-55.83, 35.0),
        # ylim=(-12.85, 22.48),
    )

    p = pearsonr(
        [i[-1].tolist() for i in gen[:]], [i[-1].tolist() for i in disc[:]]
    ).statistic
    print("Pearson: ", p)

    pyes = pearsonr(
        [i[-1].tolist() for i in f(gen[:], L, "yes")],
        [i[-1].tolist() for i in f(disc[:], L, "yes")],
    ).statistic
    pno = pearsonr(
        [i[-1].tolist() for i in f(gen[:], L, "no")],
        [i[-1].tolist() for i in f(disc[:], L, "no")],
    ).statistic
    print("Pearson for pos: ", pyes)
    print("Pearson for neg: ", pno)
    plt.grid()
    plt.plot()


def logitlens_viz(words, input_words, max_probs, savename=None):
    """
    Plot heatmap for logitlens argmax words over layers and token
    positions.
    """

    import matplotlib.colors as mcolors
    import numpy as np

    norm = mcolors.Normalize(
        vmin=np.min(max_probs.detach().cpu().numpy()),
        vmax=np.max(max_probs.detach().cpu().numpy()),
    )

    cmap = sns.color_palette("viridis", as_cmap=True)

    plt.figure(figsize=(20, 12))
    ax = sns.heatmap(
        max_probs.detach().cpu().numpy(),
        annot=np.array(words),
        fmt="",
        cmap=cmap,
        linewidths=0.5,
        cbar_kws={"label": "Log Probability"},
        norm=norm,
    )

    plt.title("Logit Lens Visualization")
    plt.xlabel("Input Tokens")
    plt.ylabel("Layers")

    plt.yticks(np.arange(len(words)) + 0.5, range(len(words)))

    plt.gca().xaxis.tick_top()
    plt.gca().xaxis.set_label_position("top")
    plt.xticks(np.arange(len(input_words)) + 0.5, input_words, rotation=45)
    # plt.show()
    if savename is not None:
        plt.savefig(savename)
    # plt.close()

def plot_logodds_over_layers(lgo):
    """
    Plot mean log-odds and standard deviation over layers across examples.
    """
    X = np.array([i.numpy() for i in lgo])
    meansg = np.mean(X, 0)
    stdsg = np.std(X, 0)
    x = list(range(len(lgo[0])))
    # x = np.arange(1, 26)
    plt.plot(x, meansg, ".-")
    plt.fill_between(x, meansg - stdsg, meansg + stdsg, color="b", alpha=0.2)
    plt.grid()
    plt.xlabel("Layer")
    plt.ylabel("log-odds")
    plt.show()
