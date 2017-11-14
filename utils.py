import numpy as np





def common_limits(ax, limits=None):
    if limits is None:
        limits = np.array([ax.get_xlim(), ax.get_ylim()])
        limits = (np.min(limits), np.max(limits))

    ax.plot(limits, limits, c="#666666", lw=1, linestyle=":", zorder=-1)

    ax.set_xlim(limits)
    ax.set_ylim(limits)
