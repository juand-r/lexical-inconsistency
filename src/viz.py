"""
Code for visualizations

TODO: add code for stacked barplot to show token ratios across layers.
"""


def f(L1, L2, c):
    """Filter subset of L1 where entries of L2 have taxonomic==c"""
    return [i for ii, i in enumerate(L1) if L2[ii].taxonomic == c]


def plot_correlations(L, gen, disc):
    """
    TODO
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
    TODO
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
