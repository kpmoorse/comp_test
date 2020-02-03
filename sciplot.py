import matplotlib.pyplot as plt

def broken_axes(ax=None):

    if not ax:
        ax = plt.gca()

    # Offset primary axes
    ax.spines["bottom"].set_position(('axes',0)) 
    ax.spines["left"].set_position(('axes',0))

    # Hide secondary axes
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')

    # Remove axis ends
    ax.spines['left'].set_bounds(*(ax.get_yticks()[[1,-2]]))
    ax.spines['bottom'].set_bounds(*(ax.get_xticks()[[1,-2]]))
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)