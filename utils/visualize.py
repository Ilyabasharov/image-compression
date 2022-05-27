import os
import random
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from MulticoreTSNE import MulticoreTSNE as TSNE

@torch.no_grad()
def plot_latent_tsne(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    save_path: str=None,
) -> None:
    
    model.eval()
    
    latent_vectors = []
    classes = []
    
    for gt_batch in tqdm(dataloader):
        z = model(
            x=gt_batch['images']['test_image'].to(next(model.parameters()).device),
            embedding=True,
        )
        z = z.detach().cpu().numpy()
        
        latent_vectors.append(z)
        classes.append(gt_batch['class'].numpy())
        
    latent_vectors = np.concatenate(latent_vectors)
    classes = np.concatenate(classes)
    
    latent_vectors_2d = TSNE(
        n_jobs=-1,
        n_iter=1000,
    ).fit_transform(latent_vectors)
    
    plt.figure(figsize=(10, 10))
    
    scatters = {}
    colors = mcolors.TABLEAU_COLORS.values()
    
    for l, c in zip(set(classes), colors):
        x = latent_vectors_2d[:, 0][classes == l]
        y = latent_vectors_2d[:, 1][classes == l]
        scatters[c] = plt.scatter(
            x=x, y=y, c=c,
        )
        
    plt.legend(
        tuple(scatters.values()), 
        tuple(set(classes)), 
        fontsize=16,
        markerscale=1.7,
        loc='best',
        ncol=1,
    )
    
    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'tsne.jpg'), format='jpg')
        
    plt.show()
    
def generate_samples_between_centers(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    save_path: str=None,
    from_class: int=9,
    to_class: int=7,
    rows_cols: int=2,
    device: torch.device='cuda',
) -> None:
    
    centroids = []
    classes = []
    
    for gt_batch in tqdm(dataloader):
        preds_batch = model(
            x=gt_batch['images']['test_image'].to(next(model.parameters()).device),
            embedding=False,
        )
        
        centroids.append(preds_batch['mu'].detach().cpu().numpy())
        classes.append(gt_batch['class'].numpy())
        
    centroids = np.concatenate(centroids)
    classes = np.concatenate(classes)
    
    center_from = centroids[classes == from_class].mean(axis=0)
    center_to = centroids[classes == to_class].mean(axis=0)
    
    z = torch.stack([
        torch.from_numpy(t*center_from + (1 - t) * center_to)
        for t in np.linspace(0, 1, rows_cols*rows_cols)
    ])
    
    images = model._decode(z.to(device)).permute(0, 2, 3, 1).cpu().detach().numpy()
    
    fig = plt.figure(figsize=(10, 10))

    for i, img in enumerate(images):
        fig.add_subplot(rows_cols, rows_cols, i + 1)
        plt.imshow(img)
    
    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'samples_between_centers.jpg'), format='jpg')
    plt.show()
    
@torch.no_grad()
def visualize_prediction(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    save_path: str=None,
    n_samples: int=10,
    device: torch.device='cuda',
) -> None:
    
    fig, axes = plt.subplots(n_samples, 3, figsize=(10, 40))

    for i in range(n_samples):
        index = random.randint(0, len(dataset))
            
        sample = dataset[index]
        
        inputs = sample['image']
        outputs = sample['test_image']
        
        if model is not None:
            model.eval()
            preds = model(
                x=inputs.to(next(model.parameters()).device).unsqueeze(0),
                embedding=False,
            )['pred_image'].squeeze(0)
        else:
            preds = torch.zeros(inputs.shape)
            
        for j, (data, title) in enumerate(zip([inputs, outputs, preds], ['input', 'output', 'pred'])):            
            axes[i][j].imshow(data.permute(1, 2, 0).cpu().numpy())
            axes[i][j].set_title(title)
    
    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'predictions.jpg'), format='jpg')
        
    plt.show()