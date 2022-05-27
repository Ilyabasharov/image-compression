import os
import collections
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler


def trainer(
    model: nn.Module,
    optimizer: nn.Module,
    scheduler: nn.Module,
    dataloaders_dict: dict,
    num_epochs: int,
    device: torch.device,
    loss_function: nn.Module,
    visualiser: object,
    where_to_save: str,
) -> None:
    
    scaler = GradScaler()
    
    min_val_loss = 1e10
    loss_per_epoch = collections.defaultdict(lambda: collections.defaultdict(list))
    
    for epoch in range(num_epochs):
        
        loss_per_batch = collections.defaultdict(lambda: collections.defaultdict(list))
        
        for phase in dataloaders_dict:
            
            phase_train = phase != 'test'
            
            model.train() if phase_train else model.eval()
            
            for gt_batch in tqdm(
                dataloaders_dict[phase],
                desc=f'{phase}, lr={scheduler.get_last_lr()[-1]:.6f}',
                total=len(dataloaders_dict[phase]),
            ):
                
                gt_batch = {
                    key: gt_batch[key].to(device)
                    for key in gt_batch
                }
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase_train), autocast():
                    
                    preds_batch = model(
                        x=gt_batch['image'] if phase_train else gt_batch['test_image'],
                        embedding=False,
                    )
                    
                    loss_batch = loss_function(gt_batch, preds_batch)
                    
                    if phase_train:
                        
                        scaler.scale(loss_batch['summary']).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
                    
                    for loss_type in loss_batch:
                        loss_per_batch[phase][loss_type].append(loss_batch[loss_type].item())
            
            for loss_type in loss_per_batch[phase]:
                loss_per_epoch[phase][loss_type].append(
                    sum(loss_per_batch[phase][loss_type]) / len(dataloaders_dict[phase].dataset)
            )
        
            if not phase_train and loss_per_epoch[phase]['summary'][-1] < min_val_loss:
                min_val_loss = loss_per_epoch[phase]['summary'][-1]
                
                # save the model
                torch.save(model.state_dict(), os.path.join(where_to_save, 'model.pth'))
            
        if visualiser is not None:
            visualiser.step(loss_per_epoch)