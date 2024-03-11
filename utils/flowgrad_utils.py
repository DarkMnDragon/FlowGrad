import functools

import torch
import torchvision
import numpy as np
import abc

from RectifiedFlow.models.utils import from_flattened_numpy, to_flattened_numpy
from scipy import integrate

import imageio

import lpips
import clip

from .DiffAugment_pytorch import DiffAugment

import os
import time


@torch.no_grad()
def embed_to_latent_rk45(model_fn, img):
    device = img.device

    def ode_func(t, x):
        x = from_flattened_numpy(x, img.shape).to(device).type(torch.float32)
        vec_t = torch.ones(img.shape[0], device=x.device) * t
        drift = model_fn(x, vec_t * 999)
        return to_flattened_numpy(drift)

    rtol = atol = 1e-5
    method = 'RK45'
    eps = 1e-3

    # Initial sample
    x = img.detach().clone()

    # NOTE: 实验添加 eps
    solution = integrate.solve_ivp(ode_func, (1., eps), to_flattened_numpy(x),
                                   rtol=rtol, atol=atol, method=method)
    nfe = solution.nfev  # Number of Function Evaluations
    num_time_steps = len(solution.t) - 1  # Number of time steps
    print(f"Embed to Latent, nfe = {nfe}, num_time_steps = {num_time_steps}")
    print("Time Steps:", solution.t)
    print("Solution Shape:", solution.y.shape)  # [flattened_img, num_time_steps]
    print("Img Shape:", img.shape)
    x = torch.tensor(solution.y[:, -1]).reshape(img.shape).to(device).type(torch.float32)

    return x


@torch.no_grad()
def embed_to_latent_leapfrog(model_fn, img, N=100):
    """
        The reversed leapfrog to recover z_0 and z_{1/2N}.
  
        Args:
          model_fn: A velocity model.
          img: z_1.
        Returns:
          z_0, z_{1/2N}: reversed latent code.
    """
    with torch.no_grad():
        z1 = img.detach.clone()
        device = z1.device

        ### Uniform
        dt = 1. / N
        eps = 1e-3  # default: 1e-3
        t = torch.ones(z1.shape[0], device=device)

        use_rk45_initialize = False
        use_euler_initialize = True
        use_exact_initialize = False

        if (use_euler_initialize):  # Method 1: Single Half Step Euler
            print("use_euler_initialize!")
            pred = model_fn(z_prev, t * 999)
            z = z_prev.detach().clone() - pred * dt / 2

        if (use_exact_initialize):  # Method 2: Exact z_{1-1/2N}
            print("use_exact_initialize!")
            z = torch.load("xxx")

        if (use_rk45_initialize):  # Method 3: rk45
            print("use_rk45_initialize!")
            rtol = atol = 1e-7
            method = 'RK45'
            eps = 1e-3

            def ode_func(t, x):
                x = from_flattened_numpy(x, img.shape).to(device).type(torch.float32)
                vec_t = torch.ones(img.shape[0], device=x.device) * t
                drift = model_fn(x, vec_t * 999)
                return to_flattened_numpy(drift)

            x = img
            solution = integrate.solve_ivp(ode_func, (1., (2. * N - 1) / (2. * N) * (1. - eps) + eps),
                                           to_flattened_numpy(x),
                                           rtol=rtol, atol=atol, method=method)
            nfe = solution.nfev  # Number of Function Evaluations
            num_time_steps = len(solution.t) - 1  # Number of time steps
            print(f"Embed to First Reverse in Leapfrog, nfe = {nfe}, num_time_steps = {num_time_steps}")
            print("Time Steps:", solution.t)
            z = torch.tensor(solution.y[:, -1]).reshape(img.shape).to(device).type(torch.float32)

        # torch.save(z,
                  #  f'/home/lrl/rectifiedflow/RectifiedFlow/ImageGeneration/logs/celebahq/eval/ckpt_10/z_{2 * sde.sample_N - 1}_rev_rk45.pt')

        for i in range(2 * N - 1, 0, -1):
            # NOTE: RL: What's the effect of the eps here ??
            num_t = i / (2. * N) * (1. - eps) + eps
            print(f"Reversed Leapfrog, step = {i}, time = {num_t:.5f}, z_{i - 1}=z_{i + 1}-f(z_{i})*dt")
            t.fill_(num_t)
            pred = model_fn(z, t * 999)  ### Copy from models/utils.py 
            z_next = z_prev - pred * dt

            # if (2 * N - i <= 10 or i <= 10) :
            #   torch.save(z_next, f'/home/lrl/rectifiedflow/RectifiedFlow/ImageGeneration/logs/celebahq/eval/ckpt_10/z_{i-1}_rev_leapfrog.pt')

            z_prev, z = z, z_next
        
        return z, z_prev


@torch.no_grad()
def generate_traj_euler(dynamic, z0, u=None, N=100, straightness_threshold=None):
    traj = []

    # Initial sample
    z = z0.detach().clone()
    traj.append(z.detach().clone().cpu())
    batchsize = z0.shape[0]

    dt = 1. / N
    eps = 1e-3
    pred_list = []
    for i in range(N):
        t = torch.ones(z0.shape[0], device=z0.device) * i / N * (1. - eps) + eps
        pred = dynamic(z, t * 999)

        if (u is not None):
            try:
                # z = z + u[i]
                pred = pred + u[i]
            except:
                pass

        # t = torch.ones(z0.shape[0], device=z0.device) * i / N * (1. - eps) + eps
        # pred = dynamic(z, t * 999)
        z = z.detach().clone() + pred * dt

        traj.append(z.detach().clone())

        pred_list.append(pred.detach().clone().cpu())

    if straightness_threshold is not None:
        ### compute straightness and construct G
        non_uniform_set = {}
        non_uniform_set['indices'] = []
        non_uniform_set['length'] = {}
        accumulate_length = 0
        accumulate_straightness = 0
        cur_index = 0
        for i in range(N):
            try:
                d1 = (pred_list[i - 1] - pred_list[i]).pow(2).sum() / pred_list[i].pow(2).sum()
            except:
                d1 = 0

            try:
                d2 = (pred_list[i + 1] - pred_list[i]).pow(2).sum() / pred_list[i].pow(2).sum()
            except:
                d2 = 0

            d = max(d1, d2)
            accumulate_straightness += d
            accumulate_length += 1
            if (accumulate_straightness > straightness_threshold) or (i == (N - 1)):
                non_uniform_set['length'][cur_index] = accumulate_length
                non_uniform_set['indices'].append(cur_index)

                accumulate_straightness = 0
                accumulate_length = 0
                cur_index = i + 1

        return traj, non_uniform_set
    else:
        return traj


# @torch.no_grad()  # NOTE: 这个函数的作用是什么 ？？
# def generate_traj_with_guidance(dynamic, z0, N=100, L=None, alpha_L=1.0):
#   traj = []

#   # Initial sample
#   z = z0.detach().clone()
#   traj.append(z.detach().clone().cpu())
#   batchsize = z0.shape[0]

#   dt = 1./N
#   eps = 1e-3
#   for i in range(N):
#     t = torch.ones(z0.shape[0], device=z0.device) * i / N * (1.-eps) + eps

#     if L is not None:
#         with torch.enable_grad():
#             inputs = z.detach().clone()
#             inputs.requires_grad = True
#             pred = dynamic(inputs, t*999)
#             #loss = L(inputs) ### NOTE: compute loss on xt
#             loss = L(inputs + pred * (1. - t.detach().clone())) ### NOTE: compute loss on x1
#             g = torch.autograd.grad(loss, inputs)[0]
#             g *= alpha_L 
#             print(i, loss.item())

#     z = z.detach().clone() + pred * dt

#     if L is not None:
#         z = z - g.detach().clone()

#     traj.append(z.detach().clone())

#   return traj

def get_img(path=None):
    img = imageio.imread(path)  ### 4-no expression
    img = img / 255.
    img = img[np.newaxis, :, :, :]
    img = img.transpose(0, 3, 1, 2)
    print('read image from:', path, 'img range:', img.min(), img.max())
    img = torch.tensor(img).float()
    img = torch.nn.functional.interpolate(img, size=256)

    return img


def save_img(img, path=None):
    torchvision.utils.save_image(img.clamp_(0.0, 1.0), os.path.join(path), nrow=16, normalize=False)


class clip_semantic_loss():
    def __init__(self, text, img, device, alpha=0.5, replicate=20, inverse_scaler=None):
        self.loss_fn_alex = lpips.LPIPS(net='alex', spatial=False).to(device)
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device)
        clip_mode = "ViT-B/32"
        self.interp_mode = 'bilinear'
        self.clip_model, _ = clip.load(clip_mode, device=device)
        self.clip_c = self.clip_model.logit_scale.exp()
        self.text_tok = clip.tokenize([text]).to(device)
        self.policy = 'color,translation,resize,cutout'
        self.replicate = 20
        self.img = img
        self.alpha = alpha
        self.inverse_scaler = inverse_scaler

    def L_N(self, x):
        sim = (self.inverse_scaler(x) - self.img).abs().mean()

        img_aug = DiffAugment(x.repeat(self.replicate, 1, 1, 1), policy=self.policy)
        img_aug = self.inverse_scaler(img_aug)
        img_aug = torch.nn.functional.interpolate(img_aug, size=224, mode=self.interp_mode)
        img_aug.sub_(self.mean[None, :, None, None]).div_(self.std[None, :, None, None])

        logits_per_image, logits_per_text = self.clip_model(img_aug, self.text_tok)
        logits_per_image = logits_per_image / self.clip_c
        concept_loss = (-1.) * logits_per_image

        return self.alpha * concept_loss.mean() + (1. - self.alpha) * sim.sum()


def flowgrad_optimization_euler(z0, u_ind, dynamic, generate_traj, N=100, L_N=None, u_init=None,
                                number_of_iterations=10, straightness_threshold=5e-3, lr=1.0):
    device = z0.device
    shape = z0.shape
    u = {}
    if u_init is None:
        for ind in u_ind:
            u[ind] = torch.zeros_like(z0).to(z0.device)
            u[ind].requires_grad = True
            u[ind].grad = torch.zeros_like(u[ind], device=u[ind].device)
    else:
        for ind in u_init.keys():
            u[ind] = u_init[ind].detach().clone().to(z0.device)

        for ind in u_ind:
            try:
                u[ind].requires_grad = True
            except:
                u[ind] = torch.zeros_like(z0).to(z0.device)
                u[ind].requires_grad = True
            u[ind].grad = torch.zeros_like(u[ind], device=u[ind].device)

    u_optimizer = torch.optim.SGD([u[key] for key in u_ind], lr=lr)  ### white black 5e-3

    print(f"Optimization start! straightness_threshold={straightness_threshold}, lr={lr}")

    ### L is supposed to be a function (ideally, a lambda expression). The output of L should a scalar.
    L_best = 1e6
    for i in range(number_of_iterations):
        print("=" * 40, "     ", f"Iteration {i + 1} start", "     ", "=" * 40)
        u_optimizer.zero_grad()

        ### get the forward simulation result and the non-uniform discretization trajectory
        ### non_uniform_set: indices and interval length (t_{j+1} - t_j)
        z_traj, non_uniform_set = generate_traj(dynamic, z0, u=u, N=N, straightness_threshold=straightness_threshold)
        print("Non uniform timesteps info:")
        print(non_uniform_set)
        print("-" * 100)
        t_s = time.time()
        ### use lambda to store \nabla L
        inputs = torch.zeros(z_traj[-1].shape, device=device)
        inputs.data = z_traj[-1].to(device).detach().clone()
        inputs.requires_grad = True
        loss = L_N(inputs)
        grad_z_prev = torch.autograd.grad(loss, inputs)[0]
        grad_z_prev = grad_z_prev.detach().clone()

        print('   Inputs: ', inputs.view(-1).detach().cpu().numpy())
        print('   L: %.6f' % loss.detach().cpu().numpy())
        print('   grad_z1: ', grad_z_prev.reshape(-1).detach().cpu().numpy())
        print('   grad_z1 max: ', torch.max(grad_z_prev))
        print("-" * 100)

        if loss.detach().cpu().numpy() < L_best:
            opt_u = {}
            for ind in u.keys():
                opt_u[ind] = u[ind].detach().clone()
            L_best = loss.detach().cpu().numpy()
        print('L_best:%.6f' % L_best)

        if i == number_of_iterations: break

        eps = 1e-3  # default: 1e-3

        for j in range(N - 1, -1, -1):
            if j in non_uniform_set['indices']:
                assert j in u_ind
            else:
                continue

            ### compute lambda: correct vjp version

            ### NOTE: modified!
            u[j].grad = grad_z_prev.detach().clone() * non_uniform_set['length'][j] / N

            inputs = torch.zeros(grad_z_prev.shape, device=device)
            inputs.data = z_traj[j].to(device).detach().clone()
            inputs.requires_grad = True
            t = (torch.ones((1,)) * j / N * (1. - eps) + eps) * 999

            # func = lambda x: (x.contiguous().reshape(shape) + u[j].detach().clone() + \
            # dynamic(x.contiguous().reshape(shape) + u[j].detach().clone(), t.detach().clone()) * non_uniform_set['length'][j] / N).view(-1)

            func = lambda x: (x.contiguous().reshape(shape) +
                              (dynamic(x.contiguous().reshape(shape), t.detach().clone()) + u[j].detach().clone())
                              * non_uniform_set['length'][j] / N).view(-1)

            output, vjp = torch.autograd.functional.vjp(func, inputs=inputs.view(-1),
                                                        v=grad_z_prev.detach().clone().reshape(-1))

            grad_z_prev = vjp.detach().clone().contiguous().reshape(shape)

            del inputs
            if j == 0: break

        print('BP time:', time.time() - t_s)
        ### Re-assignment  
        for j in range(len(non_uniform_set['indices'])):
            start = non_uniform_set['indices'][j]
            try:
                end = non_uniform_set['indices'][j + 1]
            except:
                end = N

            for k in range(start, end):
                if k in u_ind:
                    u[k].grad = u[start].grad.detach().clone()

        u_optimizer.step()

        print("=" * 40, "     ", f"Iteration {i + 1} end", "     ", "=" * 40, "\n")

    opt_u = {}
    for ind in u.keys():
        opt_u[ind] = u[ind].detach().clone()

    return opt_u
