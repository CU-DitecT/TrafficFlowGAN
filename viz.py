import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

from src.metrics_factory.get_KL import get_kde_curve
from src.utils import check_exist_and_create, load_json, save_dict_to_json

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_dir', default='experiments/lwr',
                    help="Directory containing 'test' ")
parser.add_argument('--result_dir', default='experiments/lwr/test_result/lwr/',
                    help="folder to viz")
parser.add_argument('--mode', default='debug',
                    help="mode debug keeps more detail; mode paper is clean' ")
parser.add_argument('--sudoku', default=True, action='store_true',
                    help="while to plot sudoku")
parser.add_argument('--force_overwrite', default=False, action='store_true',
                    help="For debug. Force to clean the 'figure' folder each running ")


COMPARE_HIST_SUDOKU = {"alpha":0.5,
                       "bins":20,
                       "density": True,
                       "level": 3,
                       "fontsize":16,
                       "colors": ["r", "b"],
                       "choose_lowest_kl":[True, False],
                       "width": 6,
                       "height": 4}


def compare_hist_sudoku(feature,y_true, y_pred, kl,
                        level=3, width=6, height=4,
                        mode="debug", choose_lowest_kl=False,
                        save_at=None):
    # the name of label
    if "_v" in save_at:
        label_name = "vel"
    else:
        label_name = "acc"
    # sort the y_true by its mean value/kl in an increasing order
    idx_increase_by_mean = np.argsort(np.mean(y_true, axis=1))
    idx_to_plot = np.empty((level, level), dtype=int)
    cut = y_true.shape[0] - y_true.shape[0] % level # e.g. 200 -> 198, which can be divided by level=3
    if choose_lowest_kl is False:
        idx_to_plot = idx_increase_by_mean[:cut].reshape(level, -1)[:,:level]
    elif choose_lowest_kl is True:
        idx_cut_mean = idx_increase_by_mean[:cut].reshape(level, -1)
        kl_value = kl[idx_cut_mean] # think of kl as a look-up table for kl values
        small_kl_idx = np.argsort(kl_value, axis=1)[:,:level] # return positions in table "idx_to_plot"

        for row in range(level):
            idx_to_plot[row,:] = idx_cut_mean[row, small_kl_idx[row, :]]

    fig, axs = plt.subplots(level, level, figsize=(level*width, level*height))
    for i in range(level):
        for j in range(level):
            if level>1:
                ax = axs[i,j]
            else:
                ax = axs
            idx = idx_to_plot.T[i,j]
            ax.hist(y_pred[idx, :], color=COMPARE_HIST_SUDOKU["colors"][0],
                                    alpha=COMPARE_HIST_SUDOKU["alpha"],
                                    bins=COMPARE_HIST_SUDOKU["bins"],
                                    density=COMPARE_HIST_SUDOKU["density"], label='sim')

            x_marginal, kde_marginal = get_kde_curve(y_pred[idx, :])
            ax.plot(x_marginal, kde_marginal, COMPARE_HIST_SUDOKU["colors"][0]+'-', label='sim')

            if i == level-1:
                ax.set_xlabel(label_name, fontsize=COMPARE_HIST_SUDOKU["fontsize"])

            ax.hist(y_true[idx, :], color=COMPARE_HIST_SUDOKU["colors"][1],
                                    alpha=COMPARE_HIST_SUDOKU["alpha"],
                                    bins=COMPARE_HIST_SUDOKU["bins"],
                                    density=COMPARE_HIST_SUDOKU["density"], label='real')
            x_marginal, kde_marginal = get_kde_curve(y_true[idx, :])
            ax.plot(x_marginal, kde_marginal, COMPARE_HIST_SUDOKU["colors"][1]+'-', label='real')
            ax.legend()

            if mode == "debug":
                title = f"The No.{idx:d}/{y_true.shape[0]:d}; {label_name}={np.mean(y_true[idx,:]):.3f}; kl={kl[idx]:.3f}"
                ax.set_title(title+' (for {})'.format(feature))

    plt.tight_layout()

    plt.savefig(save_at,
                dpi=300,
                bbox_inches="tight")
    plt.close()


def main(experiment_dir, result_dir, mode="debug", sudoku=True, interval="prediction",
         metrics_statistic=True,
         force_overwrite=False):
    # experiment_dir: where contains "test" and create folder "figures"
    # result_dir: where to load result folder

    # load data:
    #y_true = np.loadtxt(os.path.join(result_dir, "targets_test.csv"), delimiter=",", dtype=np.float32)
    #y_pred = np.loadtxt(os.path.join(result_dir, "predictions_test.csv"), delimiter=",", dtype=np.float32)
    #kl = np.loadtxt(os.path.join(result_dir, "kl_test.csv"), delimiter=",", dtype=np.float32)

    y_true_rho = np.loadtxt(os.path.join(result_dir, "targets_test_rho.csv"), delimiter=",", dtype=np.float32)
    y_pred_rho = np.loadtxt(os.path.join(result_dir, "predictions_test_rho.csv"), delimiter=",", dtype=np.float32)
    kl_rho = np.loadtxt(os.path.join(result_dir, "kl_rho_test.csv"), delimiter=",", dtype=np.float32)
    y_true_u = np.loadtxt(os.path.join(result_dir, "targets_test_u.csv"), delimiter=",", dtype=np.float32)
    y_pred_u = np.loadtxt(os.path.join(result_dir, "predictions_test_u.csv"), delimiter=",", dtype=np.float32)
    kl_u = np.loadtxt(os.path.join(result_dir, "kl_u_test.csv"), delimiter=",", dtype=np.float32)

    alias = os.path.basename(result_dir)
    viz_dir = os.path.join(experiment_dir, "viz", alias)
    check_exist_and_create(viz_dir)


    # sudoku
    if sudoku is True:
        for choose_lowest_kl in COMPARE_HIST_SUDOKU["choose_lowest_kl"]:
            level = COMPARE_HIST_SUDOKU["level"]
            width = COMPARE_HIST_SUDOKU["width"]
            height = COMPARE_HIST_SUDOKU["height"]
            save_at_rho = os.path.join(viz_dir,
                                   f"(rho)-mode={mode}-sudoku-level={level:d}-lowest_kl={choose_lowest_kl:d}.png")
            save_at_u = os.path.join(viz_dir,
                                   f"(u)-mode={mode}-sudoku-level={level:d}-lowest_kl={choose_lowest_kl:d}.png")
            #save_at = os.path.join(viz_dir,
            #                       f"-mode={mode}-sudoku-level={level:d}-lowest_kl={choose_lowest_kl:d}.png")
            if os.path.exists(save_at_rho) & (force_overwrite is False):
            #if os.path.exists(save_at) & (force_overwrite is False):
                print('PASS')
                pass
            else:
                compare_hist_sudoku('rho',y_true_rho, y_pred_rho, kl_rho, level=level,
                                    width=width, height=height, mode=mode,
                                    choose_lowest_kl=choose_lowest_kl,
                                    save_at=save_at_rho)
                compare_hist_sudoku('u',y_true_u, y_pred_u, kl_u, level=level,
                                    width=width, height=height, mode=mode,
                                    choose_lowest_kl=choose_lowest_kl,
                                    save_at=save_at_u)


if __name__ == "__main__":
    args = parser.parse_args()
    experiment_dir = args.experiment_dir
    result_dir = args.result_dir
    mode = args.mode
    check_exist_and_create(os.path.join(experiment_dir, "viz"))
    if result_dir is not None:
        check_folders = [result_dir]
    else:
        check_folders = [os.path.join(experiment_dir, 'test', folder) for
                         folder in os.listdir(os.path.join(experiment_dir, 'test'))]

    for check_folder in check_folders:
        main(experiment_dir, check_folder, mode=mode, sudoku=args.sudoku,
             force_overwrite=args.force_overwrite)




