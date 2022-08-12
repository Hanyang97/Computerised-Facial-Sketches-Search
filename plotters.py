"""
# @Description:
Plot utilities for face finding applications
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection as lc
from matplotlib.colors import colorConverter
from mpl_toolkits.mplot3d.art3d import Line3DCollection as lc3d
from scipy.interpolate import interp1d
import matplotlib.image as mpimg
import glob


def colored_line_segments(xs, ys, zs=None, color='k', mid_colors=False):
    if isinstance(color, str):
        color = colorConverter.to_rgba(color)[:-1]
        color = np.array([color for i in range(len(xs))])
    segs = []
    seg_colors = []
    lastColor = [color[0][0], color[0][1], color[0][2]]
    start = [xs[0], ys[0]]
    end = [xs[0], ys[0]]
    if not zs is None:
        start.append(zs[0])
        end.append(zs[0])
    else:
        zs = [zs] * len(xs)
    for x, y, z, c in zip(xs, ys, zs, color):
        if mid_colors:
            seg_colors.append([(chan + lastChan) * .5 for chan, lastChan in zip(c, lastColor)])
        else:
            seg_colors.append(c)
        lastColor = c[:-1]
        if not z is None:
            start = [end[0], end[1], end[2]]
            end = [x, y, z]
        else:
            start = [end[0], end[1]]
            end = [x, y]
        segs.append([start, end])
    colors = [(*color, 1) for color in seg_colors]
    return segs, colors


def segmented_resample(xs, ys, zs=None, color='k', n_resample=100, mid_colors=False):
    n_points = len(xs)
    if isinstance(color, str):
        color = colorConverter.to_rgba(color)[:-1]
        color = np.array([color for i in range(n_points)])
    n_segs = (n_points - 1) * (n_resample - 1)
    xsInterp = np.linspace(0, 1, n_resample)
    segs = []
    seg_colors = []
    hiResXs = [xs[0]]
    hiResYs = [ys[0]]
    if not zs is None:
        hiResZs = [zs[0]]
    RGB = color.swapaxes(0, 1)
    for i in range(n_points - 1):
        fit_xHiRes = interp1d([0, 1], xs[i:i + 2])
        fit_yHiRes = interp1d([0, 1], ys[i:i + 2])
        xHiRes = fit_xHiRes(xsInterp)
        yHiRes = fit_yHiRes(xsInterp)
        hiResXs = hiResXs + list(xHiRes[1:])
        hiResYs = hiResYs + list(yHiRes[1:])
        R_HiRes = interp1d([0, 1], RGB[0][i:i + 2])(xsInterp)
        G_HiRes = interp1d([0, 1], RGB[1][i:i + 2])(xsInterp)
        B_HiRes = interp1d([0, 1], RGB[2][i:i + 2])(xsInterp)
        lastColor = [R_HiRes[0], G_HiRes[0], B_HiRes[0]]
        start = [xHiRes[0], yHiRes[0]]
        end = [xHiRes[0], yHiRes[0]]
        if not zs is None:
            fit_zHiRes = interp1d([0, 1], zs[i:i + 2])
            zHiRes = fit_zHiRes(xsInterp)
            hiResZs = hiResZs + list(zHiRes[1:])
            start.append(zHiRes[0])
            end.append(zHiRes[0])
        else:
            zHiRes = [zs] * len(xHiRes)

        if mid_colors: seg_colors.append([R_HiRes[0], G_HiRes[0], B_HiRes[0]])
        for x, y, z, r, g, b in zip(xHiRes[1:], yHiRes[1:], zHiRes[1:], R_HiRes[1:], G_HiRes[1:], B_HiRes[1:]):
            if mid_colors:
                seg_colors.append([(chan + lastChan) * .5 for chan, lastChan in zip((r, g, b), lastColor)])
            else:
                seg_colors.append([r, g, b])
            lastColor = [r, g, b]
            if not z is None:
                start = [end[0], end[1], end[2]]
                end = [x, y, z]
            else:
                start = [end[0], end[1]]
                end = [x, y]
            segs.append([start, end])

    colors = [(*color, 1) for color in seg_colors]
    data = [hiResXs, hiResYs]
    if not zs is None:
        data = [hiResXs, hiResYs, hiResZs]
    return segs, colors, data


def faded_segment_resample(xs, ys, zs=None, color='k', fade_len=20, n_resample=100, direction='Head'):
    segs, colors, hiResData = segmented_resample(xs, ys, zs, color, n_resample)
    n_segs = len(segs)
    if fade_len > len(segs):
        fade_len = n_segs
    if direction == 'Head':
        # Head fade
        alphas = np.concatenate((np.zeros(n_segs - fade_len), np.linspace(0, 1, fade_len)))
    else:
        # Tail fade
        alphas = np.concatenate((np.linspace(1, 0, fade_len), np.zeros(n_segs - fade_len)))
    colors = [(*color[:-1], alpha) for color, alpha in zip(colors, alphas)]
    return segs, colors, hiResData


def test2d():
    NPOINTS = 10
    RESAMPLE = 10
    N_FADE = int(RESAMPLE * NPOINTS * 0.5)
    N_SEGS = (NPOINTS - 1) * (RESAMPLE - 1)

    SHOW_POINTS_AXI_12 = True
    SHOW_POINTS_AXI_34 = True

    np.random.seed(11)
    xs = np.random.rand(NPOINTS)
    ys = np.random.rand(NPOINTS)

    MARKER = '.'
    CMAP = plt.get_cmap('hsv')
    COLORS = np.array([CMAP(i)[:-1] for i in np.linspace(0, 1, NPOINTS)])
    MARKER_COLOR = COLORS

    N_SCATTER = (NPOINTS - 1) * (RESAMPLE - 1) + 1
    COLORS_LONG = np.array([CMAP(i)[:-1] for i in np.linspace(1 / N_SCATTER, 1, N_SCATTER)])

    fig = plt.figure(figsize=(12, 8), dpi=100)
    ax1 = fig.add_subplot(221)  # original data
    segs, colors = colored_line_segments(xs, ys, color=COLORS, mid_colors=True)
    if SHOW_POINTS_AXI_12: ax1.scatter(xs, ys, marker=MARKER, color=COLORS)
    ax1.add_collection(lc(segs, colors=colors))
    ax1.text(.05, 1.05, 'Original Data')
    ax1.set_ylim(0, 1.2)

    ax2 = fig.add_subplot(222, sharex=ax1, sharey=ax1)  # resampled data
    segs, colors, hiResData = segmented_resample(xs, ys, color=COLORS, n_resample=RESAMPLE)
    if SHOW_POINTS_AXI_12: ax2.scatter(hiResData[0], hiResData[1], marker=MARKER, color=COLORS_LONG)
    ax2.add_collection(lc(segs, colors=colors))
    ax2.text(.05, 1.05, 'Original Data - Resampled')
    ax2.set_ylim(0, 1.2)

    ax3 = fig.add_subplot(223, sharex=ax1, sharey=ax1)  # resampled with linear alpha fade start to finish

    segs, colors, hiResData = faded_segment_resample(xs, ys, color=COLORS, fade_len=RESAMPLE * NPOINTS, n_resample=RESAMPLE, direction='Head')
    if SHOW_POINTS_AXI_34: ax3.scatter(hiResData[0], hiResData[1], marker=MARKER, color=COLORS_LONG)
    ax3.add_collection(lc(segs, colors=colors))
    ax3.text(.05, 1.05, 'Resampled - w/Full length fade')
    ax3.set_ylim(0, 1.2)

    ax4 = fig.add_subplot(224, sharex=ax1, sharey=ax1)  # resampled with linear alpha fade N_FADE long
    segs, colors, hiResData = faded_segment_resample(xs, ys, color=COLORS, fade_len=N_FADE, n_resample=RESAMPLE, direction='Head')
    if SHOW_POINTS_AXI_34: ax4.scatter(hiResData[0], hiResData[1], marker=MARKER, color=COLORS_LONG)
    ax4.add_collection(lc(segs, colors=colors))
    ax4.text(.05, 1.05, 'Resampled - w/{} point fade'.format(N_FADE))
    ax4.set_ylim(0, 1.2)

    fig.savefig('2d_fadeSegmentedColorLine.png')
    plt.show()


def test3d():
    def set_view(axi):
        axi.set_xlim(-.65, .65)
        axi.set_ylim(-.65, .75)
        axi.set_zlim(-.65, .65)
        axi.view_init(elev=45, azim=45)

    NPOINTS = 40
    RESAMPLE = 2
    N_FADE = int(RESAMPLE * NPOINTS * 0.5)

    N_FADE = 20

    N_SEGS = (NPOINTS - 1) * (RESAMPLE - 1)

    SHOW_POINTS_AXI_12 = True
    SHOW_POINTS_AXI_34 = False

    alpha = np.linspace(.5, 1.5, NPOINTS) * np.pi
    theta = np.linspace(.25, 1.5, NPOINTS) * np.pi
    rad = np.linspace(0, 1, NPOINTS)
    xs = rad * np.sin(theta) * np.cos(alpha)
    ys = rad * np.sin(theta) * np.sin(alpha)
    zs = rad * np.cos(theta)

    MARKER = '.'
    CMAP = plt.get_cmap('hsv')
    COLORS = np.array([CMAP(i)[:-1] for i in np.linspace(0, 1, NPOINTS)])
    MARKER_COLOR = COLORS

    N_SCATTER = (NPOINTS - 1) * (RESAMPLE - 1) + 1
    COLORS_LONG = np.array([CMAP(i)[:-1] for i in np.linspace(1 / N_SCATTER, 1, N_SCATTER)])

    fig = plt.figure(figsize=(12, 8), dpi=100)
    ax1 = fig.add_subplot(221, projection='3d')  # original data
    segs, colors = colored_line_segments(xs, ys, zs, color=COLORS, mid_colors=True)
    if SHOW_POINTS_AXI_12: ax1.scatter(xs, ys, zs, marker=MARKER, color=COLORS)
    ax1.add_collection(lc3d(segs, colors=colors))

    ax2 = fig.add_subplot(222, projection='3d', sharex=ax1, sharey=ax1)  # resampled data
    segs, colors, hiResData = segmented_resample(xs, ys, zs, color=COLORS, n_resample=RESAMPLE)
    if SHOW_POINTS_AXI_12: ax2.scatter(hiResData[0], hiResData[1], hiResData[2], marker=MARKER, color=COLORS_LONG)
    ax2.add_collection(lc3d(segs, colors=colors))

    ax3 = fig.add_subplot(223, projection='3d', sharex=ax1, sharey=ax1)  # resampled with linear alpha fade start to finish
    segs, colors, hiResData = faded_segment_resample(xs, ys, zs, color=COLORS, fade_len=RESAMPLE * NPOINTS, n_resample=RESAMPLE, direction='Head')
    if SHOW_POINTS_AXI_34: ax3.scatter(hiResData[0], hiResData[1], hiResData[2], marker=MARKER, color=COLORS_LONG)
    ax3.add_collection(lc3d(segs, colors=colors))

    ax4 = fig.add_subplot(224, projection='3d', sharex=ax1, sharey=ax1)  # resampled with linear alpha fade N_FADE long
    segs, colors, hiResData = faded_segment_resample(xs, ys, zs, color=COLORS, fade_len=N_FADE, n_resample=RESAMPLE, direction='Head')
    if SHOW_POINTS_AXI_34: ax4.scatter(hiResData[0], hiResData[1], hiResData[2], marker=MARKER, color=COLORS_LONG)
    ax4.add_collection(lc3d(segs, colors=colors))

    labels = ('Original Data',
              'Original Data - Resampled',
              'Resampled - w/Full length fade',
              'Resampled - w/{} point fade'.format(N_FADE))

    for ax, label in zip((ax1, ax2, ax3, ax4), labels):
        set_view(ax)
        ax.text(.6, -.6, 1.55, label)

    fig.savefig('3d_fadeSegmentedColorLine.png')
    plt.show()


def plot_trace_2d(xs, ys, n_trace, true_theta, face_folder):
    NPOINTS = len(xs)
    SHOW_POINTS_AXI_12 = True
    MARKER = 'o'
    CMAP = plt.get_cmap('gist_heat')
    COLORS = np.array([CMAP(i)[:-1] for i in np.linspace(0, 1, NPOINTS)])
    DotCMAP = plt.get_cmap('plasma')
    DotCOLORS = np.array([DotCMAP(i)[:-1] for i in np.linspace(0, 1, NPOINTS)])

    fig = plt.figure(figsize=(6, 4), dpi=100, constrained_layout=True)
    ax1 = fig.add_subplot()
    segs, colors = colored_line_segments(xs, ys, color=COLORS, mid_colors=True)
    if SHOW_POINTS_AXI_12: ax1.scatter(xs, ys, marker=MARKER, color=DotCOLORS, alpha=1)
    ax1.add_collection(lc(segs, linewidths=1.5, linestyles=':', colors=colors))

    ax1.set_xlabel('x')
    ax1.set_xlabel('y')

    # plot true theta
    for theta in true_theta:
        ax1.scatter(theta[0], theta[1], c='r', marker='D')

    fig.savefig(os.path.join(face_folder, f"trace_{n_trace}.png"))
    plt.close()


def plot_trace_3d(xs, ys, zs, n_trace, true_theta, face_folder):
    NPOINTS = len(xs)
    SHOW_POINTS_AXI_12 = True

    MARKER = 'o'
    CMAP = plt.get_cmap('copper')
    COLORS = np.array([CMAP(i)[:-1] for i in np.linspace(0, 1, NPOINTS)])
    DotCMAP = plt.get_cmap('plasma')
    DotCOLORS = np.array([DotCMAP(i)[:-1] for i in np.linspace(0, 1, NPOINTS)])

    fig = plt.figure(figsize=(6, 4), dpi=100, constrained_layout=True)
    ax1 = fig.add_subplot(projection='3d')  # original data
    segs, colors = colored_line_segments(xs, ys, zs, color=COLORS, mid_colors=True)
    if SHOW_POINTS_AXI_12: ax1.scatter(xs, ys, zs, marker=MARKER, color=DotCOLORS, alpha=1)
    ax1.add_collection(lc3d(segs, linewidths=1.5, linestyles=':', colors=colors))

    ax1.view_init(elev=20, azim=20)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')

    # plot true theta
    for theta in true_theta:
        ax1.scatter(theta[0], theta[1], theta[2], c='r', marker='D')

    fig.savefig(os.path.join(face_folder, f"trace_{n_trace}.png"))
    plt.close()


def plot_trace(trace_i, p_dim, T, run_df, true_theta, norm=1, *args, **kwargs):
    # mk_plot_dir(f"res_dim_{p_dim}")

    if kwargs.get("face_finding"):
        xis = run_df[[f'xi_{i}' for i in range(p_dim)]]
        dist = np.linalg.norm(np.subtract(xis, true_theta), ord=norm, axis=-1)

        face_folder = kwargs.get("face_folder")
        num_imgs = len(glob.glob1(face_folder, f"target_{trace_i}*"))
        num_col = 5
        remain = num_imgs % num_col
        num_row = 1 + num_imgs // num_col if not remain else 2 + num_imgs // num_col

        fig, axs = plt.subplots(nrows=num_row, ncols=num_col, figsize=(6.4, 1.6*num_row), gridspec_kw={'height_ratios': np.ones(num_row).tolist()})

        # Plot recon images
        img_idx = 0
        for r in range(1, num_row):
            for c in range(num_col):
                if os.path.exists(os.path.join(face_folder, f'target_{trace_i}_recon_{img_idx}.png')):
                    img = mpimg.imread(os.path.join(face_folder, f'target_{trace_i}_recon_{img_idx}.png'))
                    axs[r, c].imshow(img)
                    img_idx += 5
                    axs[r, c].axes.xaxis.set_visible(False)
                    axs[r, c].axes.yaxis.set_visible(False)
                else:
                    continue

        # Plot target image at the last axis
        img = mpimg.imread(os.path.join(face_folder, f'target_{trace_i}.png'))
        axs[-1, -(num_col-remain+1)].imshow(img)
        axs[-1, -(num_col-remain+1)].axes.xaxis.set_visible(False)
        axs[-1, -(num_col-remain+1)].axes.yaxis.set_visible(False)

        # Remove redundant axises
        for ax in axs[-1, -(num_col-remain):]:
            ax.remove()

        # Plot 1d distances at top
        gs = axs[0, 0].get_gridspec()
        # remove the underlying axes
        for ax in axs[0, 0:]:
            ax.remove()
        axbig = fig.add_subplot(gs[0, :])
        axbig.plot(run_df["order"], dist, 'ro--')

        # Plot points of shown faces
        face_order = []
        face_points = []
        for p in range(len(dist)):
            if p % 5 == 0:
                face_order.append(run_df["order"][p])
                face_points.append(dist[p])
        axbig.plot(face_order, face_points, 'bo', label='displayed faces')
        axbig.set(xlabel='order', ylabel='distance', title=f'distance for p is {p_dim}')
        axbig.axes.xaxis.set_visible(False)
        axbig.grid()
        axbig.legend()

        # Save plot
        plt.savefig(os.path.join(face_folder, f"trace_{trace_i}_combo.png"))
        plt.close()
    if kwargs.get("categorical_face_finding"):
        xis = run_df[[f'xi_{i}' for i in range(p_dim)]]
        dist = np.linalg.norm(np.subtract(xis, true_theta), axis=-1)

        face_folder = kwargs.get("face_folder")
        num_imgs = len(glob.glob1(face_folder, "target_0*"))
        num_col = 5
        remain = num_imgs % num_col
        num_row = 1 + num_imgs // num_col if not remain else 2 + num_imgs // num_col

        fig, axs = plt.subplots(nrows=num_row, ncols=num_col, gridspec_kw={'height_ratios': np.ones(num_row).tolist()})

        # Plot recon images
        img_idx = 0
        for r in range(1, num_row):
            for c in range(num_col):
                if os.path.exists(os.path.join(face_folder, f'target_{trace_i}_recon_{img_idx}.png')):
                    img = mpimg.imread(os.path.join(face_folder, f'target_{trace_i}_recon_{img_idx}.png'))
                    axs[r, c].imshow(img)
                    img_idx += 5
                    axs[r, c].axes.xaxis.set_visible(False)
                    axs[r, c].axes.yaxis.set_visible(False)
                else:
                    continue

        # Plot target image at the last axis
        img = mpimg.imread(os.path.join(face_folder, f'target_{trace_i}.png'))
        axs[-1, -(num_col-remain+1)].imshow(img)
        axs[-1, -(num_col-remain+1)].axes.xaxis.set_visible(False)
        axs[-1, -(num_col-remain+1)].axes.yaxis.set_visible(False)

        # Remove redundant axises
        for ax in axs[-1, -(num_col-remain):]:
            ax.remove()

        # Plot 1d distances at top
        gs = axs[0, 0].get_gridspec()
        # remove the underlying axes
        for ax in axs[0, 0:]:
            ax.remove()
        axbig = fig.add_subplot(gs[0, :])
        axbig.plot(run_df["order"], dist, 'ko--')

        # Decorate points in 1d plot
        face_order, face_points = [], []
        green_order, green_points = [], []
        red_order, red_points = [], []
        amber_order, amber_points = [], []

        for id in range(len(dist)):
            if np.array_equal(run_df["observations"][id], [1,0,0]):     # Green
                green_order.append(run_df["order"][id])
                green_points.append(dist[id])
            elif np.array_equal(run_df["observations"][id], [0,1,0]):   # Red
                red_order.append(run_df["order"][id])
                red_points.append(dist[id])
            else:   # Amber
                amber_order.append(run_df["order"][id])
                amber_points.append(dist[id])
            if id % 5 == 0:
                face_order.append(run_df["order"][id])
                face_points.append(dist[id])
        axbig.plot(face_order, face_points, 'kD', markersize=8)
        axbig.plot(green_order, green_points, 'g.')
        axbig.plot(red_order, red_points, 'r.')
        axbig.plot(amber_order, amber_points, 'y.')

        axbig.set(xlabel='order', ylabel='distance', title=f'distance for p is {p_dim}')
        axbig.axes.xaxis.set_visible(False)
        axbig.grid()
        # plt.show()

        # Save plot
        plt.savefig(os.path.join(face_folder, f"trace_{trace_i}_combo.png"))
        plt.close()
    else:
        if p_dim == 1:
            fig, ax = plt.subplots()
            ax.plot(run_df["order"], run_df[f"xi_0"], 'ro--')
            ax.plot(T, true_theta, 'bo')
            ax.set(xlabel='order', ylabel='location')
            ax.grid()
            plt.savefig(os.path.join(face_folder, f"trace_{trace_i}.png"))
            plt.close()
        elif p_dim == 2:
            plot_trace_2d(run_df["xi_0"], run_df["xi_1"], trace_i, true_theta, face_folder)
        elif p_dim == 3:
            plot_trace_3d(run_df["xi_0"], run_df["xi_1"], run_df["xi_2"], trace_i, true_theta, face_folder)
            # Save data
            run_df.to_csv(os.path.join(face_folder, f'trace_{trace_i}.csv'))  # save trace data
            np.save(os.path.join(face_folder, f'target_{trace_i}.npy', true_theta))  # save target data
        elif p_dim > 3:
            xis = run_df[[f'xi_{i}' for i in range(p_dim)]]
            dist = np.linalg.norm(np.subtract(xis, true_theta), axis=-1, ord=norm)

            fig, ax = plt.subplots()
            ax.plot(run_df["order"], dist, 'ro--')
            ax.set(xlabel='order', ylabel='distance', title=f'Distance when p is {p_dim}')
            ax.grid()
            plt.savefig(os.path.join(face_folder, f"trace_{trace_i}.png"))
            plt.close()


def mk_plot_dir(dir_name):
    if os.path.exists(dir_name): return
    else:
        os.makedirs(dir_name)

if __name__ == "__main__":
    # # 2D plot demo
    # NPOINTS = 30
    # true_theta = np.array([[2, 2]])
    # xs = np.random.rand(NPOINTS)
    # ys = np.random.rand(NPOINTS)
    # plot_trace_2d(xs, ys, 0, true_theta)
    #
    # # 3D plot demo
    # NPOINTS = 30
    # true_theta = np.array([[0, 0, 0]])
    # alpha = np.linspace(.5, 1.5, NPOINTS) * np.pi
    # theta = np.linspace(.25, 1.5, NPOINTS) * np.pi
    # rad = np.linspace(0, 1, NPOINTS)
    # xs = rad * np.sin(theta) * np.cos(alpha)
    # ys = rad * np.sin(theta) * np.sin(alpha)
    # zs = rad * np.cos(theta)
    # plot_trace_3d(xs, ys, zs, 1, true_theta)


    # <editor-fold desc="[+] Face plot">
    # fig, axs = plt.subplots(nrows=4, ncols=5, figsize=(10,6))
    fig, axs = plt.subplots(nrows=4, ncols=5, gridspec_kw={'height_ratios': [1,1,1,1]})
    gs = axs[0, 0].get_gridspec()
    # remove the underlying axes
    for ax in axs[0, 1:]:
        ax.remove()
    axbig = fig.add_subplot(gs[0, :])
    axbig.annotate('Big Axes \nGridSpec[1:, -1]', (0.1, 0.5),
                   xycoords='axes fraction', va='center')

    fig.tight_layout()

    plt.show()
    # </editor-fold>