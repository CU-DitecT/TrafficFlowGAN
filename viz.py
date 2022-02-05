import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from plotting import newfig, savefig
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata

from src.metrics_factory.get_KL import get_kde_curve
from src.utils import check_exist_and_create, load_json, save_dict_to_json

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_dir', default='experiments/ngsim_learning_z',
                    help="Directory containing 'test' ")
parser.add_argument('--result_dir', default='experiments/ngsim_learning_z/test_result/ngsim_learning_z/', 
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

def plot_pred_data(y_pred_variable,Exact_variable,variable,viz_dir,X_star,x,t,X,T,idx):
    #y_pred_variable: y_pred_rho or y_pred_u
    #Exact_variable:Exact_rho or Exact_u
    #variable: 'rho' or 'u'

    save_at= os.path.join(viz_dir,variable)

    ###row 0
    fig, ax = newfig(1.0, 1.1)
    ax.axis('off')
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])
   
    VARIABLE_pred = np.mean(y_pred_variable, axis = 1)
    VARIABLE_pred=griddata(X_star, VARIABLE_pred.flatten(), (X, T), method='cubic')
    Sigma_pred_variable = np.var(y_pred_variable, axis = 1)
    Sigma_pred_variable = griddata(X_star, Sigma_pred_variable, (X, T), method='cubic')

    variable_star = Exact_variable.flatten()[:,None]
    variable_train = variable_star[idx,:]
    X_variable_train = X_star[idx,:]
    h = ax.imshow(VARIABLE_pred.T, interpolation='nearest', cmap='rainbow', 
              extent=[t.min(), t.max(), x.min(), x.max()], 
              origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    ax.plot(X_variable_train[:,1:2], X_variable_train[:,0:1], 'kx', label = 'Data (%d points)' % (variable_train.shape[0]), markersize = 1, clip_on = False)

    # line = np.linspace(x.min(), x.max(), 2)[:,None]
    # ax.plot(t[25]*np.ones((2,1)), line, 'w-', linewidth = 1)
    # ax.plot(t[50]*np.ones((2,1)), line, 'w-', linewidth = 1)
    # ax.plot(t[75]*np.ones((2,1)), line, 'w-', linewidth = 1)    
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.legend(frameon=False, loc = 'best')
    ax.set_title('${}(t,x)$'.format(variable), fontsize = 10)
    plt.savefig(save_at+'_data',
                dpi=300,
                bbox_inches="tight")
    plt.close()

    ###row 1
    fig, ax = newfig(1.0)
    ax.axis('off')
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1-1/3, bottom=0, left=0.1, right=0.9, wspace=0.5)

    id1=25
    ax = plt.subplot(gs1[0, 0])
    ax.plot(x,Exact_variable[id1,:], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,VARIABLE_pred[id1,:], 'r--', linewidth = 2, label = 'Prediction')
    lower = VARIABLE_pred[id1,:] - 2.0*np.sqrt(Sigma_pred_variable[id1,:])
    upper = VARIABLE_pred[id1,:] + 2.0*np.sqrt(Sigma_pred_variable[id1,:])
    plt.fill_between(x.flatten(), lower.flatten(), upper.flatten(), 
                 facecolor='orange', alpha=0.5, label="Two std band")
    ax.set_xlabel('$x$')
    ax.set_ylabel('${}(t,x)$'.format(variable))    
    ax.set_title('$t = 0.078$', fontsize = 10)
    ax.axis('square')
    ax.set_xlim([-0.1,1.1])
    ax.set_ylim([-0.1,1.1])

    id2=75
    ax = plt.subplot(gs1[0, 1])
    ax.plot(x,Exact_variable[id2,:], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,VARIABLE_pred[id2,:], 'r--', linewidth = 2, label = 'Prediction')
    lower = VARIABLE_pred[id2,:] - 2.0*np.sqrt(Sigma_pred_variable[id2,:])
    upper = VARIABLE_pred[id2,:] + 2.0*np.sqrt(Sigma_pred_variable[id2,:])
    plt.fill_between(x.flatten(), lower.flatten(), upper.flatten(), 
                 facecolor='orange', alpha=0.5, label="Two std band")
    ax.set_xlabel('$x$')
    ax.set_ylabel('${}(t,x)$'.format(variable))
    ax.axis('square')
    #ax.set_xlim([-0.1,1.1])
    #ax.set_ylim([-0.1,1.1])
    ax.set_title('$t = 0.234$', fontsize = 10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)

    id3=-1
    ax = plt.subplot(gs1[0, 2])
    ax.plot(x,Exact_variable[id3,:], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,VARIABLE_pred[id3,:], 'r--', linewidth = 2, label = 'Prediction')
    lower = VARIABLE_pred[id3,:] - 2.0*np.sqrt(Sigma_pred_variable[id3,:])
    upper = VARIABLE_pred[id3,:] + 2.0*np.sqrt(Sigma_pred_variable[id3,:])
    plt.fill_between(x.flatten(), lower.flatten(), upper.flatten(), 
                     facecolor='orange', alpha=0.5, label="Two std band")
    ax.set_xlabel('$x$')
    ax.set_ylabel('${}(t,x)$'.format(variable))
    ax.axis('square')
    #ax.set_xlim([-0.1,1.1])
    #ax.set_ylim([-0.1,1.1])     
    ax.set_title('$t = 1.0$', fontsize = 10)
    plt.savefig(save_at+'_prediction',
                    dpi=300,
                    bbox_inches="tight")
    plt.close()

    ###row 2
    fig, ax = newfig(1.0)
    ax.axis('off')
    gs2 = gridspec.GridSpec(1, 2)
    gs2.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs2[:, :])

    h = ax.imshow(Sigma_pred_variable.T, interpolation='nearest', cmap='rainbow', 
                  extent=[t.min(), t.max(), x.min(), x.max()], 
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.legend(frameon=False, loc = 'best')
    ax.set_title('Variance of ${}(t,x)$'.format(variable), fontsize = 10)
    plt.savefig(save_at+'_uncertainty ',
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

    x = np.loadtxt(os.path.join(result_dir, "x.csv"), delimiter=",", dtype=np.float32)
    t = np.loadtxt(os.path.join(result_dir, "t.csv"), delimiter=",", dtype=np.float32)
    X, T = np.meshgrid(x,t) # each is 960 by 241
    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None])) # hstack is column wise stack, 241*960 (after flatten) by 2
    
    Exact_rho= np.loadtxt(os.path.join(result_dir, "Exact_rho.csv"), delimiter=",", dtype=np.float32)
    Exact_u= np.loadtxt(os.path.join(result_dir, "Exact_u.csv"), delimiter=",", dtype=np.float32)
    idx= np.loadtxt(os.path.join(result_dir, "idx.csv"), delimiter=",", dtype=int)

    y_true_rho = np.loadtxt(os.path.join(result_dir, "targets_test_rho.csv"), delimiter=",", dtype=np.float32)
    y_pred_rho = np.loadtxt(os.path.join(result_dir, "predictions_test_rho.csv"), delimiter=",", dtype=np.float32)
    kl_rho = np.loadtxt(os.path.join(result_dir, "kl_rho_test.csv"), delimiter=",", dtype=np.float32)
    y_true_u = np.loadtxt(os.path.join(result_dir, "targets_test_u.csv"), delimiter=",", dtype=np.float32)
    y_pred_u = np.loadtxt(os.path.join(result_dir, "predictions_test_u.csv"), delimiter=",", dtype=np.float32)
    kl_u = np.loadtxt(os.path.join(result_dir, "kl_u_test.csv"), delimiter=",", dtype=np.float32)

    alias = os.path.basename(result_dir)
    viz_dir = os.path.join(experiment_dir, "viz", alias)
    check_exist_and_create(viz_dir)

    #plt1
    plot_pred_data(y_pred_rho,Exact_rho,'rho',viz_dir,X_star,x,t,X,T,idx)
    plot_pred_data(y_pred_u,Exact_u,'u',viz_dir,X_star,x,t,X,T,idx)



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
            save_at = os.path.join(viz_dir,
                                 f"-mode={mode}-sudoku-level={level:d}-lowest_kl={choose_lowest_kl:d}.png")
            if os.path.exists(save_at_rho) & (force_overwrite is False):
            ##if os.path.exists(save_at) & (force_overwrite is False):
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




