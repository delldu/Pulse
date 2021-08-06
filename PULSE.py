from stylegan import G_synthesis, G_mapping
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import time
import torch
from loss import LossBuilder
import pdb

import math
from torch.optim import Optimizer

# Spherical Optimizer Class
# Uses the first two dimensions as batch information
# Optimizes over the surface of a sphere using the initial radius throughout
#
# Example Usage:
# opt = SphericalOptimizer(torch.optim.SGD, [x], lr=0.01)


class SphericalOptimizer(Optimizer):
    def __init__(self, optimizer, params, **kwargs):
        self.opt = optimizer(params, **kwargs)
        self.params = params
        with torch.no_grad():
            self.radii = {
                param: (
                    param.pow(2).sum(tuple(range(2, param.ndim)), keepdim=True) + 1e-9
                ).sqrt()
                for param in params
            }
        # (Pdb) optimizer
        # <class 'torch.optim.adam.Adam'>
        # (Pdb) len(params), params[0].size(), params[5].size()
        # (6, [1, 18, 512], [1, 1, 16, 16])

    @torch.no_grad()
    def step(self, closure=None):
        loss = self.opt.step(closure)
        for param in self.params:
            param.data.div_(
                (
                    param.pow(2).sum(tuple(range(2, param.ndim)), keepdim=True) + 1e-9
                ).sqrt()
            )
            param.mul_(self.radii[param])

        return loss


class PULSE(torch.nn.Module):
    def __init__(self, cache_dir, verbose=True):
        super(PULSE, self).__init__()

        self.synthesis = G_synthesis().cuda()
        self.verbose = verbose

        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        if self.verbose:
            print("Loading Synthesis Network")

        # with open_url("https://drive.google.com/uc?id=1TCViX1YpQyRsklTVYEJwdbmK91vklCo8", cache_dir=cache_dir, verbose=verbose) as f:
        #     self.synthesis.load_state_dict(torch.load(f))

        self.synthesis.load_state_dict(torch.load("cache/synthesis.pt"))

        for param in self.synthesis.parameters():
            param.requires_grad = False

        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.2)

        self.gaussian_fit = torch.load("cache/gaussian_fit.pt")
        # mapping = G_mapping().cuda()

        # with open_url("https://drive.google.com/uc?id=14R6iHGf5iuVx3DMNsACAl7eBr7Vdpd0k", cache_dir=cache_dir, verbose=verbose) as f:
        #         mapping.load_state_dict(torch.load(f))

        # if self.verbose: print("\tRunning Mapping Network")
        # with torch.no_grad():
        #     torch.manual_seed(0)
        #     latent = torch.randn((1000000,512),dtype=torch.float32, device="cuda")
        #     latent_out = torch.nn.LeakyReLU(5)(mapping(latent))
        #     self.gaussian_fit = {"mean": latent_out.mean(0), "std": latent_out.std(0)}
        #     torch.save(self.gaussian_fit,"gaussian_fit.pt")

    def forward(
        self,
        ref_im,
        seed,
        loss_str,
        eps,
        noise_type,
        num_trainable_noise_layers,
        tile_latent,
        bad_noise_layers,
        opt_name,
        learning_rate,
        steps,
        lr_schedule,
        **kwargs,
    ):

        # loss_str = '100*L2+0.05*GEOCROSS'
        # eps = 0.002
        # noise_type = 'trainable'
        # num_trainable_noise_layers = 5
        # bad_noise_layers = '17'
        # opt_name = 'adam'
        # learning_rate = 0.4
        # steps = 100
        # lr_schedule = 'linear1cycledrop'
        # kwargs = {'input_dir': 'input', 'output_dir': 'runs', 'cache_dir': 'cache', 'duplicates': 1, 'batch_size': 1}

        tile_latent = True

        # seed = None
        if seed:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True

        batch_size = ref_im.shape[0]
        # batch_size == 1
        # (Pdb) ref_im.size() -- [1, 3, 32, 32]

        # Generate latent tensor
        # tile_latent = False
        if tile_latent:
            latent = torch.randn(
                (batch_size, 1, 512),
                dtype=torch.float,
                requires_grad=True,
                device="cuda",
            )
        else:
            latent = torch.randn(
                (batch_size, 18, 512),
                dtype=torch.float,
                requires_grad=True,
                device="cuda",
            )

        # Generate list of noise tensors
        noise = []  # stores all of the noise tensors
        noise_vars = []  # stores the noise tensors that we want to optimize on

        # noise_type == 'trainable'
        for i in range(18):
            # dimension of the ith noise tensor
            res = (batch_size, 1, 2 ** (i // 2 + 2), 2 ** (i // 2 + 2))

            new_noise = torch.randn(res, dtype=torch.float, device="cuda")
            # num_trainable_noise_layers == 5
            if i < num_trainable_noise_layers:
                new_noise.requires_grad = True
                noise_vars.append(new_noise)
            else:
                new_noise.requires_grad = False

            noise.append(new_noise)

        # (Pdb) len(noise), noise[0].size(), noise[17].size()
        # 18, [1, 1, 4, 4], [1, 1, 1024, 1024]
        # (Pdb) noise[0]
        # tensor([[[[-0.4076, -1.8516,  1.2239,  0.2655],
        #           [ 0.4495, -0.4564, -1.4785, -0.4633],
        #           [ 0.2427,  1.7014, -1.2897,  0.5059],
        #           [ 0.8207, -0.1593,  0.0294, -1.7026]]]], device='cuda:0',
        #        requires_grad=True)
        # (Pdb) noise[17]
        # tensor([[[[0., 0., 0.,  ..., 0., 0., 0.],
        #           [0., 0., 0.,  ..., 0., 0., 0.],
        #           [0., 0., 0.,  ..., 0., 0., 0.],
        #           ...,
        #           [0., 0., 0.,  ..., 0., 0., 0.],
        #           [0., 0., 0.,  ..., 0., 0., 0.],
        #           [0., 0., 0.,  ..., 0., 0., 0.]]]], device='cuda:0')

        var_list = [latent] + noise_vars

        # (Pdb) len(var_list) -- 6
        # (Pdb) len(noise_vars), noise_vars[0].size(), noise_vars[4].size()
        # (5, [1, 1, 4, 4]), [1, 1, 16, 16])

        # opt_name = 'adam'
        opt_dict = {
            "sgd": torch.optim.SGD,
            "adam": torch.optim.Adam,
            "adamax": torch.optim.Adamax,
        }
        opt_func = opt_dict[opt_name]
        # torch.optim.Adam ?
        opt = SphericalOptimizer(opt_func, var_list, lr=learning_rate)
        # (Pdb) latent
        # tensor([[[-0.1946, -0.6408, -1.8464,  ..., -0.1906, -0.1616, -0.6535],
        #          [-0.6044, -0.4074, -0.1534,  ..., -1.2470, -0.4122, -0.1046],
        #          [ 1.4960, -0.1268,  0.0833,  ..., -0.4061, -1.4328,  0.8896],
        #          ...,
        #          [-0.2607,  0.0134, -1.5494,  ...,  0.3231, -1.7157, -0.2732],
        #          [-1.3181, -1.3402, -1.3557,  ..., -0.3245, -0.1104, -0.7225],
        #          [-1.9875,  0.0337, -1.9513,  ...,  1.2133, -1.0213,  1.1432]]],
        #        device='cuda:0', requires_grad=True)

        schedule_dict = {
            "fixed": lambda x: 1,
            "linear1cycle": lambda x: (9 * (1 - np.abs(x / steps - 1 / 2) * 2) + 1)
            / 10,
            "linear1cycledrop": lambda x: (
                9 * (1 - np.abs(x / (0.9 * steps) - 1 / 2) * 2) + 1
            )
            / 10
            if x < 0.9 * steps
            else 1 / 10 + (x - 0.9 * steps) / (0.1 * steps) * (1 / 1000 - 1 / 10),
        }
        # lr_schedule = 'linear1cycledrop'
        schedule_func = (
            lambda x: (9 * (1 - np.abs(x / (0.9 * steps) - 1 / 2) * 2) + 1) / 10
            if x < 0.9 * steps
            else 1 / 10 + (x - 0.9 * steps) / (0.1 * steps) * (1 / 1000 - 1 / 10)
        )
        # schedule_dict[lr_schedule]
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt.opt, schedule_func)

        loss_builder = LossBuilder(ref_im, loss_str, eps).cuda()

        min_loss = np.inf
        min_l2 = np.inf
        best_summary = ""
        start_t = time.time()
        gen_im = None

        if self.verbose:
            print("Optimizing")
        for j in range(steps):
            opt.opt.zero_grad()

            # Duplicate latent in case tile_latent = True
            # tile_latent = False
            if tile_latent:
                latent_in = latent.expand(-1, 18, -1)
            else:
                latent_in = latent

            # Apply learned linear mapping to match latent distribution to that of the mapping network
            latent_in = self.lrelu(
                latent_in * self.gaussian_fit["std"] + self.gaussian_fit["mean"]
            )

            # Normalize image to [0,1] instead of [-1,1]

            gen_im = (self.synthesis(latent_in, noise) + 1) / 2

            # Calculate Losses
            loss, loss_dict = loss_builder(latent_in, gen_im)
            loss_dict["TOTAL"] = loss

            # Save best summary for log
            if loss < min_loss:
                min_loss = loss
                best_summary = f"BEST ({j+1}) | " + " | ".join(
                    [f"{x}: {y:.4f}" for x, y in loss_dict.items()]
                )
                best_im = gen_im.clone()

            loss_l2 = loss_dict["L2"]

            if loss_l2 < min_l2:
                min_l2 = loss_l2

            loss.backward()
            opt.step()
            scheduler.step()

        total_t = time.time() - start_t
        current_info = f" | time: {total_t:.1f} | it/s: {(j+1)/total_t:.2f} | batchsize: {batch_size}"
        if self.verbose:
            print(best_summary + current_info)
        if min_l2 <= eps:
            yield (
                gen_im.clone().cpu().detach().clamp(0, 1),
                loss_builder.D(best_im).cpu().detach().clamp(0, 1),
            )
        else:
            print("Could not find a face that downscales correctly within epsilon")
