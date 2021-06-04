#%%
import os
import pandas as pd
import scipy.stats as stats
import matplotlib as mpl
import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results

plt.style.use('seaborn-notebook')
width = 600  # Get this from LaTeX using \showthe\textwidth
nice_fonts = {
              # Use LaTeX to write all text
              "text.usetex": True,
              "font.family": "Latin Modern Roman",
              # Use 10pt font in plots, to match 10pt font in document
              "axes.labelsize": 11,
              "font.size": 11,
              # Make the legend/label fonts a little smaller
              "legend.fontsize": 8,
              "xtick.labelsize": 8,
              "ytick.labelsize": 8,
              }
mpl.rcParams.update(nice_fonts)
# path = '/home/nirnai/Cloud/Uni/TUM/MasterThesis/LaTex/figures'

def set_size(width, fraction=1):
    """ 
    Set aesthetic figure dimensions to avoid scaling in latex.
    Parameters
    ----------
    width: float
        Width in pts
    fraction: float
        Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple Dimensions of figure in inches
    """
    # Width of figure
    fig_width_pt = width * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27
    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**.5 - 1) / 2
    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio
    fig_dim = (fig_width_in, fig_height_in)
    return fig_dim

pkl_dir = '/Users/yuqichengzhu/Desktop/DeepRL-master/20210601/logs/optimizer'
pkl_paths = [os.path.join(pkl_dir, path) for path in os.listdir(pkl_dir) if path.endswith('.pkl')]
fig, ax = plt.subplots(1,1,figsize=set_size(width))
for log in pkl_paths:
    data = pd.read_pickle(log)
    means = data.mean(axis=1)
    se = stats.sem(data, axis=1)
    low, high = stats.t.interval(0.95, data.shape[1]-1 ,loc=means, scale=se)
    x = [i for i in range(data.shape[0])]
    ax.plot(x, means, label=log.split('_')[-1][:-4])
    ax.fill_between(x, low, high, alpha=0.2)
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Reward')
plt.title('The impact of modifying optimizer in PPO for CartPole-v1')
plt.show()
