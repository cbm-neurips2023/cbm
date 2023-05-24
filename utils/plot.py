import os
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from utils.utils import to_numpy


def set_axes_equal(ax):
    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)


def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])


# ---------------------------------- TO CREATE A SERIES OF PICTURES ---------------------------------- #
# from https://zulko.wordpress.com/2012/09/29/animate-your-3d-plots-with-pythons-matplotlib/

def make_views(ax, angles, elevation=None, width=4, height=3,
               prefix='tmprot_', **kwargs):
    """
    Makes jpeg pictures of the given 3d ax, with different angles.
    Args:
        ax (3D axis): te ax
        angles (list): the list of angles (in degree) under which to
                       take the picture.
        width,height (float): size, in inches, of the output images.
        prefix (str): prefix for the files created.

    Returns: the list of files created (for later removal)
    """

    files = []
    ax.figure.set_size_inches(width, height)

    for i, angle in enumerate(angles):
        ax.view_init(elev=elevation, azim=angle)
        fname = '%s%03d.jpeg' % (prefix, i)
        ax.figure.savefig(fname)
        files.append(fname)

    return files


# ----------------------- TO TRANSFORM THE SERIES OF PICTURE INTO AN ANIMATION ----------------------- #

def make_movie(files, output, fps=10, bitrate=1800, **kwargs):
    """
    Uses mencoder, produces a .mp4/.ogv/... movie from a list of
    picture files.
    """

    output_name, output_ext = os.path.splitext(output)
    command = {'.mp4': 'mencoder "mf://%s" -mf fps=%d -o %s.mp4 -ovc lavc\
                         -lavcopts vcodec=msmpeg4v2:vbitrate=%d'
                       % (",".join(files), fps, output_name, bitrate)}

    command['.ogv'] = command['.mp4'] + '; ffmpeg -i %s.mp4 -r %d %s' % (output_name, fps, output)

    print(command[output_ext])
    output_ext = os.path.splitext(output)[1]
    os.system(command[output_ext])


def make_gif(files, output, delay=100, repeat=True, **kwargs):
    """
    Uses imageMagick to produce an animated .gif from a list of
    picture files.
    """

    loop = -1 if repeat else 0
    os.system('convert -delay %d -loop %d %s %s' % (delay, loop, " ".join(files), output))


def make_strip(files, output, **kwargs):
    """
    Uses imageMagick to produce a .jpeg strip from a list of
    picture files.
    """

    os.system('montage -tile 1x -geometry +0+0 %s %s' % (" ".join(files), output))


# ---------------------------------------------- MAIN FUNCTION ---------------------------------------------- #

def rotanimate(ax, angles, output, **kwargs):
    """
    Produces an animation (.mp4,.ogv,.gif,.jpeg,.png) from a 3D plot on
    a 3D ax

    Args:
        ax (3D axis): the ax containing the plot of interest
        angles (list): the list of angles (in degree) under which to
                       show the plot.
        output : name of the output file. The extension determines the
                 kind of animation used.
        **kwargs:
            - width : in inches
            - heigth: in inches
            - framerate : frames per second
            - delay : delay between frames in milliseconds
            - repeat : True or False (.gif only)
    """

    output_ext = os.path.splitext(output)[1]

    files = make_views(ax, angles, **kwargs)

    D = {'.mp4': make_movie,
         '.ogv': make_movie,
         '.gif': make_gif,
         '.jpeg': make_strip,
         '.png': make_strip}

    D[output_ext](files, output, **kwargs)

    for f in files:
        os.remove(f)


def plot_dynamics_mask(params, inference, writer, step):
    adjacency = inference.get_adjacency()
    intervention_mask = inference.get_intervention_mask()
    if adjacency is None or intervention_mask is None:
        return
    thre = inference.get_threshold()
    adjacency = to_numpy(adjacency)
    intervention_mask = to_numpy(intervention_mask)
    adjacency_intervention = np.concatenate([adjacency, intervention_mask], axis=-1)

    obs_keys = params.obs_keys
    obs_spec = params.obs_spec
    feature_dim, action_dim = intervention_mask.shape

    fig = plt.figure(figsize=((feature_dim + action_dim) * 0.45 + 2, feature_dim * 0.45 + 2))

    vmax = thre
    while vmax <= 0.1:
        vmax = vmax * 10
        adjacency_intervention = adjacency_intervention * 10
    sns.heatmap(adjacency_intervention, linewidths=1, vmin=0, vmax=vmax, square=True, annot=True, fmt='.2f', cbar=False)

    ax = plt.gca()
    ax.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=True)
    if params.encoder_params.encoder_type == "identity":
        tick_loc = []
        cum_idx = 0
        for k in obs_keys:
            obs_dim = obs_spec[k].shape[0]
            tick_loc.append(cum_idx + obs_dim * 0.5)
            cum_idx += obs_dim
            plt.vlines(cum_idx, ymin=0, ymax=feature_dim, colors='blue', linewidths=3)
            if k != obs_keys[-1]:
                plt.hlines(cum_idx, xmin=0, xmax=feature_dim + action_dim, colors='blue', linewidths=3)

        plt.xticks(tick_loc + [feature_dim + 0.5 * action_dim], obs_keys + ["action"], rotation=90)
        plt.yticks(tick_loc, obs_keys, rotation=0)
    else:
        plt.vlines(feature_dim, ymin=0, ymax=feature_dim, colors='blue', linewidths=3)
        plt.xticks([0.5 * feature_dim, feature_dim + 0.5 * action_dim], ["feature", "action"], rotation=90)
    fig.tight_layout()
    writer.add_figure("dynamics mask", fig, step + 1)
    plt.close("all")


def plot_reward_mask(params, reward_predictor, writer, step):
    mask = reward_predictor.get_mask()
    if mask is None:
        return
    mask = to_numpy(mask)

    obs_keys = params.obs_keys
    obs_spec = params.obs_spec
    feature_dim = len(mask)

    fig = plt.figure(figsize=(feature_dim * 0.45 + 2, 0.45 + 2))

    vmax = reward_predictor.get_threshold()
    while vmax <= 0.1:
        vmax = vmax * 10
        mask = mask * 10
    sns.heatmap(mask[None], linewidths=1, vmin=0, vmax=vmax, square=True, annot=True, fmt='.2f', cbar=False)

    ax = plt.gca()
    ax.tick_params(axis="x", bottom=True, labelbottom=True)
    tick_loc = []
    cum_idx = 0
    for k in obs_keys:
        obs_dim = obs_spec[k].shape[0]
        tick_loc.append(cum_idx + obs_dim * 0.5)
        cum_idx += obs_dim
        plt.vlines(cum_idx, ymin=0, ymax=feature_dim, colors='blue', linewidths=3)

    plt.xticks(tick_loc, obs_keys, rotation=90)
    fig.tight_layout()
    writer.add_figure("reward mask", fig, step + 1)
    plt.close("all")


def plot_partition(params, partition, writer, step):
    mask = partition.get_prob()
    mask = to_numpy(mask)               # (feature_dim, num_partitions)

    obs_keys = params.obs_keys
    obs_spec = params.obs_spec
    feature_dim, num_partitions = mask.shape

    if num_partitions == 2:
        vmax = 0.5
    else:
        vmax = mask.max(axis=-1).min()

    fig = plt.figure(figsize=(feature_dim * 0.45 + 2, num_partitions * 0.45 + 2))
    sns.heatmap(mask.T, linewidths=1, vmin=0, vmax=vmax, square=True, annot=True, fmt='.2f', cbar=False)

    ax = plt.gca()
    ax.tick_params(axis="x", bottom=True, labelbottom=True)
    tick_loc = []
    cum_idx = 0
    for k in obs_keys:
        obs_dim = obs_spec[k].shape[0]
        tick_loc.append(cum_idx + obs_dim * 0.5)
        cum_idx += obs_dim
        plt.vlines(cum_idx, ymin=0, ymax=feature_dim, colors='blue', linewidths=3)
    plt.xticks(tick_loc, obs_keys, rotation=90)

    tick_loc = np.arange(num_partitions) + 0.5
    if num_partitions == 2:
        plt.yticks(tick_loc, ["relevant", "irrelevant"], rotation=0)
    elif num_partitions == 3:
        plt.yticks(tick_loc, ["x", "y", "z"], rotation=0)
    else:
        raise NotImplementedError

    fig.tight_layout()
    writer.add_figure("partition", fig, step + 1)
    plt.close("all")


def plot_abstraction(params, policy, writer, step):
    if not policy.use_abstraction:
        return
    mask = to_numpy(policy.abstraction_mask)

    obs_keys = params.obs_keys + params.goal_keys
    obs_spec = params.obs_spec
    feature_dim = len(mask)

    fig = plt.figure(figsize=(feature_dim * 0.45 + 2, 0.45 + 2))
    sns.heatmap(mask[None], linewidths=1, vmin=0, vmax=1, square=True, annot=True, fmt='.2f', cbar=False)

    ax = plt.gca()
    ax.tick_params(axis="x", bottom=True, labelbottom=True)
    tick_loc = []
    cum_idx = 0
    for k in obs_keys:
        obs_dim = obs_spec[k].shape[0]
        tick_loc.append(cum_idx + obs_dim * 0.5)
        cum_idx += obs_dim
        plt.vlines(cum_idx, ymin=0, ymax=feature_dim, colors='blue', linewidths=3)

    plt.xticks(tick_loc, obs_keys, rotation=90)
    fig.tight_layout()
    writer.add_figure("policy abstraction", fig, step + 1)
    plt.close("all")
