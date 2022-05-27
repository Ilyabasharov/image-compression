from torch import einsum
from einops import rearrange


class VectorQuantizer2(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """
    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(self, n_e, e_dim, beta, remap=None, unknown_index="random",
                 sane_index_shape=False, legacy=False):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed+1
            print(f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        match = (inds[:,:,None]==used[None,None,...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2)<1
        if self.unknown_index == "random":
            new[unknown]=torch.randint(0,self.re_embed,size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]: # extra token
            inds[inds>=self.used.shape[0]] = 0 # simply set to zero
        back=torch.gather(used[None,:][inds.shape[0]*[0],:], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach()-z)**2) + \
                   torch.mean((z_q - z.detach()) ** 2)
        else:
            loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
                   torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0],-1) # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1,1) # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[0], z_q.shape[2], z_q.shape[3])

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            indices = indices.reshape(shape[0],-1) # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1) # flatten again

        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q
        
class HilbertVAE(VariationalAE):
    def __init__(self, downlayers, uplayers, hidden_dim=512, latent_dim=128, vae=None, p=1, n=128):
        super().__init__(downlayers, uplayers, hidden_dim=512, latent_dim=128)
        self.p = p
        self.n = n
        self.hcurve = HilbertCurve(p, n)
        self.vae = vae
        
    def s_inv(self, x):
        return norm.cdf(x.detach().cpu().numpy())

    # from [0, 1]^H to Hilbert points (stertch x from [0, 1] to [0, 7])
    def proj_to_im_f(self, x : np.array):
        return (x * (2**self.p - 1)).astype(int).clip(0, 2**self.p - 1)

    # from Hilbert points (batch of [0, 2^p - 1]^H; e.g. [64, 128]) to Hilbert distances ([batch size])
    def f_inv(self, points : np.array):
        return self.hcurve.distances_from_points(points)


    # from Hilbert distances to Hilbert points
    def f(self, distances):
        return self.hcurve.points_from_distances(distances)

    # from Hilbert points to R^H
    # points : batch of [0, 2^p - 1]^H
    def s(self, points):
        return norm.ppf(np.clip(np.array(points), a_min=1e-10, a_max=None) / 2**self.p)

    def hilbert_compress(self, x):
        out = self.s_inv(x)
        out = self.proj_to_im_f(out)
        out = self.f_inv(out)

        return out

    def hilbert_compress_inv(self, distances):

        out = self.f(distances)
        out = self.s(out)

        return out
    
    def _encode(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        
        x = self.vae.down(x)
        latent_repr = self.vae.agg_down(x)
        
        latent_mu = self.vae.mu_repr(latent_repr)
        latent_log_sigma = self.vae.log_sigma_repr(latent_repr)
        
        sample = self.vae._reparametrize(latent_mu, latent_log_sigma).to(torch.float32)
        z = self.hilbert_compress(sample)
        
        return z, latent_mu, latent_log_sigma
    
    def _decode(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        
        x = self.hilbert_compress_inv(x)
        x = torch.from_numpy(x).to(torch.float32).to(device)
        x = self.vae.agg_up(x)
        x = self.vae.up(x)
        
        return x
    
    
    def forward(
        self,
        x: torch.Tensor,
        embedding: bool=False,
    ) -> torch.Tensor:
        
        z, latent_mu, latent_log_sigma = self._encode(x.to(device))
        
        if embedding:
            return z
        
        image = self._decode(z)

        return {
            'pred_image': image,
            'mu': latent_mu,
            'log_sigma': latent_log_sigma,
        }
