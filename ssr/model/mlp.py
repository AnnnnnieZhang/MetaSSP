import torch
from torch import nn
import numpy as np

from .embed import get_embedder
from ssr.ssr_utils.utils import repeat_interleave
from FFC.PlugKDD import CorNetBlock
from FFC.PlugEAA import EfficientAdditiveAttention
from FFC.PlugKAN import KAN
# from FFC.PlugMLLA import MLLABlock


class ImplicitNetwork(nn.Module):
    def __init__(
            self,
            config,
            feature_vector_size,
            sdf_bounding_sphere,
            d_in,
            d_out,
            dims,
            geometric_init=True,
            bias=1.0,
            skip_in=(),
            weight_norm=True,
            multires=0,
            sphere_scale=1.0,
            inside_outside=False
    ):
        super().__init__()

        self.use_global_encoder = config['model']['latent_feature']['use_global_encoder']
        self.use_cls_encoder = config['model']['latent_feature']['use_cls_encoder']
        self.use_sdf_encoder = config['pseudo_label']['train_protofile']

        self.sdf_bounding_sphere = sdf_bounding_sphere
        self.sphere_scale = sphere_scale
        # dims = [d_in] + dims + [d_out + feature_vector_size]
        dims = [d_in] + dims + [int(feature_vector_size/2)]

        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn

            # use cat architecture
            dims[0] = input_ch + 256                            # 39(x position encoder) + 256(pixel align feature)
            if self.use_global_encoder:
                dims[0] = dims[0] + 256                             # + 256(global img feature)
            if self.use_cls_encoder:
                dims[0] = dims[0] + 9                               # + 9(cls feature).
            if self.use_sdf_encoder:
                dims[0] = dims[0] + 256
                
            skip_dim = skip_in[0]                                   # only one skip
            dims[skip_dim] = dims[0]

        print(multires, dims)
        self.num_layers = len(dims)                             # 9
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):

            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)

                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)
        # 选择设备 (GPU 如果可用，否则 CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.decoder = KAN(
            layers_hidden = [int(feature_vector_size/2),int(feature_vector_size/8),int(feature_vector_size/16),int(feature_vector_size/32),d_out],
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0
        ).to(device)
        self.device = device


    def forward(self, input, latent_feature, cat_feature, sdf_latent):
        if self.embed_fn is not None:
            input = self.embed_fn(input)

        x = input
        num_repeats = x.shape[0] // cat_feature.shape[0]
        cat_feature = repeat_interleave(cat_feature, num_repeats)
        sdf_repeats = x.shape[0] // sdf_latent.shape[0]
        sdf_latent = repeat_interleave(sdf_latent, sdf_repeats)
        now_sampler_points = x.shape[0] // latent_feature.shape[0]
        latent_feature = repeat_interleave(latent_feature, now_sampler_points)

        x = torch.cat([x,cat_feature,latent_feature,sdf_latent], dim=1)
        x = x.to(self.device)  # 确保输入数据在 GPU 上
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:

                x = x + skip_feature

            if l == 0:
                # x = torch.cat([x, latent_feature], dim=1)

                skip_feature = x
            
            if l == self.num_layers - 2:
                feature = x

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)
    
        x = self.decoder(x)

        x = torch.cat([x,feature],dim=1)
        
        return x

    def gradient(self, x, latent_feature, cat_feature, sdf_latent):
        x.requires_grad_(True)
        y = self.forward(x, latent_feature, cat_feature, sdf_latent)[:,:1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients

    def get_outputs(self, x, latent_feature, cat_feature, sdf_latent):
        x.requires_grad_(True)                      ###### get gradients
        output = self.forward(x, latent_feature, cat_feature, sdf_latent)
        sdf = output[:,:1]
        ''' Clamping the SDF with the scene bounding sphere, so that all rays are eventually occluded '''
        if self.sdf_bounding_sphere > 0.0:
            sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - x.norm(2,1, keepdim=True))
            sdf = torch.minimum(sdf, sphere_sdf)
        feature_vectors = output[:, 1:]
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        return sdf, feature_vectors, gradients

    def get_sdf_vals(self, x, latent_feature, cat_feature, sdf_latent):
        sdf = self.forward(x, latent_feature, cat_feature, sdf_latent)[:,:1]
        ''' Clamping the SDF with the scene bounding sphere, so that all rays are eventually occluded '''
        if self.sdf_bounding_sphere > 0.0:
            sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - x.norm(2,1, keepdim=True))
            sdf = torch.minimum(sdf, sphere_sdf)
        return sdf


class RenderingNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            mode,
            d_in,
            d_out,
            dims,
            weight_norm=True,
            multires_view=0,
            per_image_code = False
    ):
        super().__init__()

        self.mode = mode
        dims = [d_in + feature_vector_size] + dims + [d_out]

        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        self.per_image_code = per_image_code
        if self.per_image_code:
            # nerf in the wild parameter
            # parameters
            # maximum 1024 images
            self.embeddings = nn.Parameter(torch.empty(1024, 32))
            std = 1e-4
            self.embeddings.data.uniform_(-std, std)
            dims[0] += 32

        print("rendering network architecture:")
        print(dims)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, points, normals, view_dirs, feature_vectors, indices):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'nerf':
            rendering_input = torch.cat([view_dirs, feature_vectors], dim=-1)
        else:
            raise NotImplementedError

        if self.per_image_code:
            image_code = self.embeddings[indices].expand(rendering_input.shape[0], -1)
            rendering_input = torch.cat([rendering_input, image_code], dim=-1)
            
        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)
        
        x = self.sigmoid(x)
        return x
    

class KANLayer(nn.Module):
    def __init__(self, in_dim, out_dim, memory_dim, num_heads):
        """
        KAN Layer with memory-enhanced attention and MLP
        Args:
            in_dim (int): Input dimension.
            out_dim (int): Output dimension.
            memory_dim (int): Memory vector dimension.
            num_heads (int): Number of attention heads.
        """
        super().__init__()
        self.memory = nn.Parameter(torch.randn(memory_dim), requires_grad=True)
        self.fc_memory = nn.Linear(memory_dim, in_dim)
        self.attention = nn.MultiheadAttention(embed_dim=in_dim, num_heads=num_heads, batch_first=True)
        self.fc_out = nn.Linear(in_dim, out_dim)
        self.activation = nn.ELU()

    def forward(self, x):
        batch_size = x.size(0)
        memory_state = self.fc_memory(self.memory).unsqueeze(0).expand(batch_size, -1, -1)
        x = x.unsqueeze(1)  # Expand for sequence dimension
        attn_output, _ = self.attention(x, x, x + memory_state)
        x = self.activation(self.fc_out(attn_output.squeeze(1)))
        return x