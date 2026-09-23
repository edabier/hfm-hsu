import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter

from src.utils import losses, utils

def compute_metrics_and_plot(E_hat=None, A_hat=None, A_gt=None, E_gt=None, model_name=None, return_results=False, plot_A=True, plot_E=True, cmap='viridis', save_mat=None):
    """
    Displays the predicted endmembers and abundances
    """
    n_graph = None

    sad = losses.SADLoss()
    mse = nn.MSELoss(reduction="sum")

    bg_colors = ["mediumpurple", "cornflowerblue", "indianred", "goldenrod", "mediumseagreen", "lightpink"]
    colors = ["thistle", "lavender", "mistyrose", "lightyellow", "lightblue", "lavenderblush"]

    if E_gt != None:
        E_gt = E_gt.detach().cpu()
        E_hat = E_hat.detach().cpu()
        A_hat = A_hat.detach().cpu()
        if E_gt.dim() == 3:
            E_gt = E_gt[0]

        E_hat, A_hat, indices = utils.order_endmembers(E_gt.T, E_hat.T , A_hat)
        E_hat = E_hat.T

    if E_hat != None:

        E_hat = E_hat.detach().cpu()
        if E_hat.dim() == 3:   
            E_hat = E_hat[0]

        c = E_hat.shape[1]
        n_graph = c // 2
        if c % 2 != 0: n_graph = n_graph + 1
            
        if E_gt != None:
            
            E_hat = utils.normalise(E_hat, is_endmember=True)
            E_gt = utils.normalise(E_gt, is_endmember=True)

            total_sad = sad(E_gt, E_hat)

            if plot_E:
                fig, axes = plt.subplots(2, n_graph, figsize=(7,5))
                axes = axes.flatten()
                for i in range(c):
                    ax = axes[i]

                    sad_val = sad(E_gt[:, i], E_hat[:, i])

                    ax.plot(E_gt[:, i], 'r', linewidth=1.0, label='GT')
                    ax.plot(E_hat[:, i], 'k-', linewidth=1.0, label='predict')
                    ax.set_title(f"SAD = {format(sad_val, '.2f')}", fontsize=10, pad=10, backgroundcolor=bg_colors[indices[i].item()], color=colors[indices[i].item()])

                    if i == 0:
                        ax.legend() 

                for j in range(i + 1, len(axes)):
                    axes[j].axis('off')

                plt.subplots_adjust(hspace=0.5, wspace=0.4)

                if model_name != None:
                    E_title = f"{model_name} Endmember estimation, SAD = {format(total_sad, '.3f')}"
                else:
                    E_title = f"Endmember estimation, SAD = {format(total_sad, '.3f')}"
                    
                plt.suptitle(E_title)

                if save_mat is not None:
                    os.makedirs(save_mat, exist_ok=True)
                    plt.savefig(f"{save_mat}/E_hat.png", bbox_inches='tight', dpi=300)
            
        else:
            if plot_E:
                if model_name != None:
                    E_title = f'{model_name} Endmember estimation'
                else:
                    E_title = 'Endmember estimation'
                for i in range(c):
                    ax = plt.subplot(2, n_graph, i + 1)
                    plt.plot(E_hat[:, i].detach().cpu(), 'k-', linewidth=1.0, label='predict')
                plt.suptitle(E_title)

                if save_mat is not None:
                    os.makedirs(save_mat, exist_ok=True)
                    plt.savefig(f"{save_mat}/E_hat.png", bbox_inches='tight', dpi=300)

    if A_gt is not None:
        
        A_gt = A_gt.detach().cpu()
        A_hat = A_hat.detach().cpu()
        if A_hat.dim() == 4:
            A_hat = A_hat[0]
        
        if A_gt.dim() == 4:
            A_gt = A_gt[0]

        if n_graph is None:
            c = A_hat.shape[0]
            n_graph = c // 2
            if c % 2 != 0: n_graph = n_graph + 1
        
        total_sad_a = sad(A_gt.flatten(1).T, A_hat.flatten(1).T)

        A_hat = utils.normalise(A_hat)
        A_gt = utils.normalise(A_gt)
        
        total_mse = mse(A_gt, A_hat)/(torch.norm(A_gt)**2)

        if plot_A:
            fig, axes = plt.subplots(2, c, figsize=(10, 5))
            for i in range(c):
                pred = axes[0, i].imshow(A_hat[i].detach().cpu(), cmap=cmap)

                mse_val = mse(A_gt[i], A_hat[i])/(torch.norm(A_gt[i])**2)

                axes[0, i].set_title(f"NMSE = {format(mse_val, '.2f')}", fontsize=10, pad=10, backgroundcolor=bg_colors[indices[i].item()], color=colors[indices[i].item()])
                axes[0, i].axis('off')

                gt = axes[1, i].imshow(A_gt[i].detach().cpu(), cmap=cmap)
                axes[1, i].axis('off')

                fig.colorbar(pred, ax=axes[0, i], fraction=0.046, pad=0.04)
                fig.colorbar(gt, ax=axes[1, i], fraction=0.046, pad=0.04)
            fig.text(0.05, 0.7, 'prediction', va='center', ha='center', fontsize=12, rotation='vertical')
            fig.text(0.05, 0.4, 'gt', va='center', ha='center', fontsize=12, rotation='vertical')
            plt.subplots_adjust(left=0.1, right=0.9, top=0.5, bottom=0.1, wspace=0.1, hspace=0.5)
            fig.tight_layout(rect=[0.05, 0.25, 0.95, 0.9])

            if model_name != None:
                A_title = f"{model_name} abundance estimation, NMSE = {format(total_mse, '.3f')}"
            else:
                A_title = f"Abundance estimation, NMSE = {format(total_mse, '.3f')}"
            plt.suptitle(A_title)

            if save_mat is not None:
                os.makedirs(save_mat, exist_ok=True)
                plt.savefig(f"{save_mat}/A_hat.png", bbox_inches='tight', dpi=300)

    elif plot_A:
        A_hat = A_hat.detach().cpu()
        if A_hat.dim() == 4:
            A_hat = A_hat[0]

        c = A_hat.shape[0]

        fig, axes = plt.subplots(1, c, figsize=(10, 5))
        for i in range(c):
            pred = axes[i].imshow(A_hat[i].detach().cpu(), cmap=cmap)
            axes[i].axis('off')
            fig.colorbar(pred, ax=axes[i], fraction=0.046, pad=0.04)
        
        # Adaptative placement of the title as a function of the number of endmembers
        offset_y = -(0.1/3) * c + 0.9
        plt.suptitle("Abundance estimation", y=offset_y)

        if save_mat is not None:
            os.makedirs(save_mat, exist_ok=True)
            plt.savefig(f"{save_mat}/A_hat.png", bbox_inches='tight', dpi=300)
    
    if return_results:
        return total_sad, total_sad_a, total_mse
    
def plot_pca_features(features, title=None):
    """
    Computes PCA on the D channels of the features and display the first 3 components
    features must be of shape (D, alpha, alpha) or (D, alpha**2)
    """

    if features.dim() > 3:
        features = features.squeeze(0)
    elif features.dim() == 2:
        features = utils.oneD_to_2d(features)
    D, alpha, _ = features.shape
    features = features.flatten(1)
    
    # alpha = int(features.shape[1]**0.5)
    X = features.T

    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X.detach().cpu().numpy())

    X_pca_reshaped = X_pca.reshape(alpha, alpha, 3).transpose(2, 0, 1)
    X_normalised = (X_pca_reshaped - X_pca_reshaped.min()) / (X_pca_reshaped.max() - X_pca_reshaped.min())

    im = plt.imshow(X_normalised.transpose(1, 2, 0))
    plt.axis('off')
    
    if title != None:
        plt.title(title)
    else:
        plt.title("PCA reduced features")
    plt.show()

def visualise_abundances(methods, dataset_name, base_path="/SOTA_results", device=None, save=False, cmap="viridis"):

    if dataset_name == "samson":
        materials = ["Soil", "Trees", "Water"]
    elif dataset_name == "jasper":
        materials = ["Trees", "Water", "Soil", "Road"]
    elif dataset_name == "apex":
        materials = ["Road", "Trees", "Roof", "Water"]
    elif dataset_name == "urban":
        materials = ["Asphalt", "Grass", "Tree", "Roof", "Metal", "Dirt"]
    else:
        print(f"Unknown dataset {dataset_name}")
        return

    # Load ground truth
    gt_A_path = f"{base_path}/GT/{dataset_name}_A.pt"
    gt_E_path = f"{base_path}/GT/{dataset_name}_E.pt"
    if not os.path.exists(gt_A_path):
        raise FileNotFoundError(f"Ground truth abundances not found at {gt_A_path}")
    gt_A = torch.tensor(torch.load(gt_A_path, map_location=device, weights_only=False), dtype=torch.float).cpu().numpy()
    gt_E = torch.tensor(torch.load(gt_E_path, map_location=device, weights_only=False), dtype=torch.float).cpu().numpy()

    # Prepare data dictionaries
    all_A = {}
    all_E = {}

    # Load method results
    for method in methods:
        A_path = f"{base_path}/{method}/{dataset_name}_A.pt"
        E_path = f"{base_path}/{method}/{dataset_name}_E.pt"
        if not os.path.exists(A_path):
            raise FileNotFoundError(f"Abundances not found for method {method} at {A_path}")
        method_A = torch.tensor(torch.load(A_path, map_location=device, weights_only=False), dtype=torch.float).cpu()
        method_E = torch.tensor(torch.load(E_path, map_location=device, weights_only=False), dtype=torch.float).cpu()
        method_E, method_A, _ = utils.order_endmembers(torch.from_numpy(gt_E).T, method_E.T, method_A)
        all_A[method] = method_A.numpy()
        all_E[method] = method_E.T.numpy()
        
    all_A['GT'] = gt_A
    all_E['GT'] = gt_E
    N = len(all_A)
    c = len(materials)

    # Find global min/max for colorbar
    vmin = min([v.min() for v in all_A.values()])
    vmax = max([v.max() for v in all_A.values()])
    
    fig = plt.figure(figsize=(3 * N, 3 * c))
    gs = gridspec.GridSpec(
        c + 1, N + 1,
        width_ratios=[0.05] + [1] * N,  # First column (material labels) is narrow
        height_ratios=[0.05] + [1] * c,  # First row (method labels) is short
        wspace=0.05,
        hspace=0.05
    )

    axes = [[fig.add_subplot(gs[i, j]) for j in range(N + 1)] for i in range(c + 1)]
    axes[0][0].axis('off')

    # Add method names to first row
    for j, method_name in enumerate(all_A.keys()):
        ax = axes[0][j + 1]
        if method_name == "DOFA" or method_name == "HyperSIGMA" or method_name == "HyperSL" or method_name == "SpecAware" or method_name == "UniverSat" or method_name == "SpecViT" or method_name == "Panopticon":
            ax.text(0.5, 1.5, method_name, ha='center', va='center', fontsize=28, backgroundcolor="#FFF28C")
        else:
            ax.text(0.5, 1.5, method_name, ha='center', va='center', fontsize=28)
        ax.axis('off')

    # Add material names to first column
    for i, material in enumerate(materials):
        ax = axes[i + 1][0]
        ax.text(-1, 0.5, material, ha='center', va='center', rotation=90, fontsize=28)
        ax.axis('off')

    # Plot abundances
    for i, material_idx in enumerate(range(c)):
        for j, method_name in enumerate(all_A.keys()):
            ax = axes[i + 1][j + 1]
            im = ax.imshow(
                all_A[method_name][material_idx, :, :],
                vmin=vmin,
                vmax=vmax,
                aspect='equal',
                cmap=cmap
            )
            ax.axis('off')

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    coloriser = mpl.colorizer.Colorizer(norm=norm, cmap=cmap)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(mpl.colorizer.ColorizingArtist(coloriser), cax=cbar_ax)
    cbar.set_ticks([0.2, 0.4, 0.6, 0.8])
    cbar.ax.tick_params(labelsize=20)

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    if save:
        plt.savefig(f"{dataset_name}_a_comparison.pdf", bbox_inches='tight')
    plt.show()

def visualise_endmembers(methods, dataset_name, base_path="/SOTA_results", wavelengths_path=None, device=None, save=False):

    if dataset_name == "samson":
        materials = ["Soil", "Trees", "Water"]
    elif dataset_name == "jasper":
        materials = ["Trees", "Water", "Soil", "Road"]
    elif dataset_name == "apex":
        materials = ["Road", "Trees", "Roof", "Water"]
    elif dataset_name == "urban":
        materials = ["Asphalt", "Grass", "Tree", "Roof", "Metal", "Dirt"]
    else:
        print(f"Unknown dataset {dataset_name}")
        return
    
    with open(wavelengths_path, "r") as file:
        lines = file.readlines()
    wavelengths = [float(line.strip()) for line in lines if line.strip()]

    gt_E_path = f"{base_path}/GT/{dataset_name}_E.pt"
    if not os.path.exists(gt_E_path):
        raise FileNotFoundError(f"Ground truth endmembers not found at {gt_E_path}")
    gt_E = utils.normalise(torch.tensor(torch.load(gt_E_path, map_location=device, weights_only=False), dtype=torch.float).cpu(), is_endmember=True).numpy()

    all_E = {}
    for method in methods:
        E_path = f"{base_path}/{method}/{dataset_name}_E.pt"
        if not os.path.exists(E_path):
            raise FileNotFoundError(f"Endmembers not found for method {method} at {E_path}")

        method_E = utils.normalise(torch.tensor(torch.load(E_path, map_location=device, weights_only=False), dtype=torch.float), is_endmember=True).cpu()
        method_E, _ = utils.order_endmembers(torch.from_numpy(gt_E).T, method_E.T)
        all_E[method] = method_E.T.numpy()

    N = len(methods)
    c = len(materials)
    num_bands = gt_E.shape[0]

    fig = plt.figure(figsize=(3 * N, 2*c))
    gs = gridspec.GridSpec(
        c + 1, N + 1,
        width_ratios=[0.03] + [1] * N,  # First column (material labels) is very narrow
        height_ratios=[0.03] + [1] * c,  # First row (method labels) is very short
        wspace=0.3,
        hspace=0.2
    )

    axes = [[fig.add_subplot(gs[i, j]) for j in range(N + 1)] for i in range(c + 1)]
    axes[0][0].axis('off')

    # Add method names to first row
    for j, method_name in enumerate(methods):
        ax = axes[0][j + 1]
        if method_name == "DOFA" or method_name == "HyperSIGMA" or method_name == "HyperSL" or method_name == "SpecAware" or method_name == "UniverSat" or method_name == "SpecViT" or method_name == "Panopticon":
            ax.text(0.5, 0, method_name, ha='center', va='center', fontsize=24, backgroundcolor="#FFF28C")
        else:
            ax.text(0.5, 0, method_name, ha='center', va='center', fontsize=24)
        ax.axis('off')

    # Add material names to first column
    for i, material in enumerate(materials):
        ax = axes[i + 1][0]
        ax.text(-6, 0.5, material, ha='center', va='center', rotation=90, fontsize=24)
        ax.axis('off')

    # Plot endmember with GT overlay
    for i, material_idx in enumerate(range(c)):
        for j, method_name in enumerate(methods):
            ax = axes[i + 1][j + 1]

            ax.plot(wavelengths, all_E[method_name][:, material_idx], 'xkcd:tangerine', linewidth=2, label=r"$\widehat{E}$")

            ax.plot(wavelengths, gt_E[:, material_idx], 'xkcd:cool blue', linewidth=2, label="GT")

            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

            if i == c - 1:
                ax.set_xlabel('Wavelength [nm]', fontsize=12)
            if j == 0:
                ax.set_ylabel('Reflectance', fontsize=12)
            if i == 0 and j == 0:
                ax.legend()

    if save:
        plt.savefig(f"{dataset_name}_e_comparison.pdf", bbox_inches='tight')
    plt.show()