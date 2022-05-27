import torch
import torch.nn as nn

class AELoss(nn.Module):
    
    def __init__(
        self,
        weights: list=[1, 1],
    ) -> None:
        
        super().__init__()
        
        self.weights = weights
        self.im2im = nn.L1Loss()
        
    def forward(
        self,
        batch_in: dict,
        batch_out: dict,
    ) -> torch.Tensor:
        
        im2im = self.im2im(batch_in['test_image'], batch_out['pred_image'])
        latent = torch.norm(batch_out['latent_repr'], p=1, dim=1).mean()
        
        loss = self.weights[0]*im2im + \
               self.weights[1]*latent
        
        return {
            'im2im': im2im,
            'latent': latent,
            'summary': loss,
        }
        
        
class KLDLoss(nn.Module):
    '''
    see Appendix B from VAE paper:
    Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    https://arxiv.org/abs/1312.6114
    0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    '''
    
    def __init__(self):
        
        super().__init__()
    
    def forward(
        self,
        mu: torch.Tensor,
        log_sigma: torch.Tensor,
    ) -> torch.Tensor:
        
        val = -0.5 * (1 + 2*log_sigma - mu**2 - (2*log_sigma).exp())
        
        return val.sum(-1).mean()
    
    
class VAELoss(nn.Module):
    '''
    KLDLoss + BCELoss
    '''
    
    def __init__(
        self,
        weights: list=[1, 1],
    ) -> None:
        
        super().__init__()
        
        self.weights = weights
        self.kld = KLDLoss()
        self.im2im = nn.BCEWithLogitsLoss(reduction='mean')
    
    def forward(
        self,
        batch_in: dict,
        batch_out: dict,
    ) -> torch.Tensor:
        
        kld = self.kld(batch_out['mu'], batch_out['log_sigma'])
        im2im = self.im2im(batch_out['pred_image'], batch_in['test_image'])
        
        loss = self.weights[0]*kld + \
               self.weights[1]*im2im
        
        return {
            'latent': kld,
            'im2im': im2im,
            'summary': loss,
        }