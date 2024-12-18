# #!/usr/bin/env python
# # coding: utf-8
# """
# Training with GradNorm Algorithm
# """

# import numpy as np
# import torch
# from LibMTL.weighting import AbsWeighting
# import torch.nn as nn
# import torch.nn.functional as F

# def gradNorm(net, layer, alpha, dataloader, num_epochs, lr1, lr2, log=False):
#     """
#     Args:
#         net (nn.Module): a multitask network with task loss
#         layer (nn.Module): a layers of the full network where appling GradNorm on the weights
#         alpha (float): hyperparameter of restoring force
#         dataloader (DataLoader): training dataloader
#         num_epochs (int): number of epochs
#         lr1（float): learning rate of multitask loss
#         lr2（float): learning rate of weights
#         log (bool): flag of result log
#     """
#     # init log
#     if log:
#         log_weights = []
#         log_loss = []
#     # set optimizer
#     optimizer1 = torch.optim.Adam(net.parameters(), lr=lr1)
#     # start traning
#     iters = 0
#     net.train()
#     for epoch in range(num_epochs):
#         # load data
#         for data in dataloader:
#             # cuda
#             if next(net.parameters()).is_cuda:
#                 data = [d.cuda() for d in data]
#             # forward pass
#             loss = net(*data)
#             # initialization
#             if iters == 0:
#                 # init weights
#                 weights = torch.ones_like(loss)
#                 weights = torch.nn.Parameter(weights)
#                 T = weights.sum().detach() # sum of weights
#                 # set optimizer for weights
#                 optimizer2 = torch.optim.Adam([weights], lr=lr2)
#                 # set L(0)
#                 l0 = loss.detach()
#             # compute the weighted loss
#             weighted_loss = weights @ loss
#             # clear gradients of network
#             optimizer1.zero_grad()
#             # backward pass for weigthted task loss
#             weighted_loss.backward(retain_graph=True)
#             # compute the L2 norm of the gradients for each task
#             gw = []
#             for i in range(len(loss)):
#                 dl = torch.autograd.grad(weights[i]*loss[i], layer.parameters(), retain_graph=True, create_graph=True)[0]
#                 gw.append(torch.norm(dl))
#             gw = torch.stack(gw)
#             # compute loss ratio per task
#             loss_ratio = loss.detach() / l0
#             # compute the relative inverse training rate per task
#             rt = loss_ratio / loss_ratio.mean()
#             # compute the average gradient norm
#             gw_avg = gw.mean().detach()
#             # compute the GradNorm loss
#             constant = (gw_avg * rt ** alpha).detach()
#             gradnorm_loss = torch.abs(gw - constant).sum()
#             # clear gradients of weights
#             optimizer2.zero_grad()
#             # backward pass for GradNorm
#             gradnorm_loss.backward()
#             # log weights and loss
#             if log:
#                 # weight for each task
#                 log_weights.append(weights.detach().cpu().numpy().copy())
#                 # task normalized loss
#                 log_loss.append(loss_ratio.detach().cpu().numpy().copy())
#             # update model weights
#             optimizer1.step()
#             # update loss weights
#             optimizer2.step()
#             # renormalize weights
#             weights = (weights / weights.sum() * T).detach()
#             weights = torch.nn.Parameter(weights)
#             optimizer2 = torch.optim.Adam([weights], lr=lr2)
#             # update iters
#             iters += 1
#     # get logs
#     if log:
#         return np.stack(log_weights), np.stack(log_loss)
    
# class GradNorm(AbsWeighting):
#     r"""Gradient Normalization (GradNorm).
    
#     This method is proposed in `GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks (ICML 2018) <http://proceedings.mlr.press/v80/chen18a/chen18a.pdf>`_ \
#     and implemented by us.

#     Args:
#         alpha (float, default=1.5): The strength of the restoring force which pulls tasks back to a common training rate.

#     """
#     def __init__(self):
#         super(GradNorm, self).__init__()
        
#     def init_param(self):
#         self.loss_scale = nn.Parameter(torch.tensor([1.0]*self.task_num, device=self.device))
        
#     def backward(self, losses, **kwargs):
#         alpha = kwargs['alpha']
#         if self.epoch >= 1:
#             loss_scale = self.task_num * F.softmax(self.loss_scale, dim=-1)
#             grads = self._get_grads(losses, mode='backward')
#             if self.rep_grad:
#                 per_grads, grads = grads[0], grads[1]
                
#             G_per_loss = torch.norm(loss_scale.unsqueeze(1)*grads, p=2, dim=-1)
#             G = G_per_loss.mean(0)
#             L_i = torch.Tensor([losses[tn].item()/self.train_loss_buffer[tn, 0] for tn in range(self.task_num)]).to(self.device)
#             r_i = L_i/L_i.mean()
#             constant_term = (G*(r_i**alpha)).detach()
#             L_grad = (G_per_loss-constant_term).abs().sum(0)
#             L_grad.backward()
#             loss_weight = loss_scale.detach().clone()
            
#             if self.rep_grad:
#                 self._backward_new_grads(loss_weight, per_grads=per_grads)
#             else:
#                 self._backward_new_grads(loss_weight, grads=grads)
#             return loss_weight.cpu().numpy()
#         else:
#             loss = torch.mul(losses, torch.ones_like(losses).to(self.device)).sum()
#             loss.backward()
#             return np.ones(self.task_num)   

# from functools import cache, partial

import torch
# import torch.distributed as dist
# from torch.autograd import grad
import torch.nn.functional as F
# from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList, Parameter

# from einops import rearrange, repeat

# from accelerate import Accelerator

# from beartype import beartype
# from beartype.door import is_bearable
# from beartype.typing import Optional, Union, List, Dict, Tuple, NamedTuple

# # helper functions

# def exists(v):
#     return v is not None

# def default(v, d):
#     return v if exists(v) else d

# # tensor helpers

# def l1norm(t, dim = -1):
#     return F.normalize(t, p = 1, dim = dim)

# # distributed helpers

# @cache
# def is_distributed():
#     return dist.is_initialized() and dist.get_world_size() > 1

# def maybe_distributed_mean(t):
#     if not is_distributed():
#         return t

#     dist.all_reduce(t)
#     t = t / dist.get_world_size()
#     return t

# main class
class DWA(Module):
    r"""Dynamic Weight Average (DWA).
    
    This method is proposed in `End-To-End Multi-Task Learning With Attention (CVPR 2019) <https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_End-To-End_Multi-Task_Learning_With_Attention_CVPR_2019_paper.pdf>`_ \
    and implemented by modifying from the `official PyTorch implementation <https://github.com/lorenmt/mtan>`_. 

    Args:
        T (float, default=2.0): The softmax temperature.

    """
    def __init__(self, device, task_num=2, num_epochs=100):
        super(DWA, self).__init__()
        self.train_loss_buffer = torch.zeros((task_num, num_epochs))
        self.task_num = task_num
        self.device = device
        
    def go(self, losses, **kwargs):
        losses = torch.stack(losses)
        # losses = losses.to(self.device)
        self.epoch = kwargs['epoch']
        T = kwargs['T']
        self.train_loss_buffer[:,self.epoch] = losses
        if self.epoch > 1:
            w_i = torch.Tensor(self.train_loss_buffer[:,self.epoch-1]/self.train_loss_buffer[:,self.epoch-2]).to(self.device)
            # print(self.epoch, w_i)
            batch_weight = self.task_num*F.softmax(w_i/T, dim=-1)
        else: #epoch 0,1
            
            # batch_weight = torch.Tensor([0.999,0.001]).to(self.device)
            batch_weight = torch.ones_like(losses).to(self.device)
        loss = torch.mul(losses, batch_weight.detach()).sum()
        loss.backward()
        return True
        # return batch_weight.detach().cpu().numpy()

# class GradNormLossWeighter(Module):
#     @beartype
#     def __init__(
#         self,
#         *,
#         num_losses: Optional[int] = None,
#         loss_weights: Optional[Union[
#             List[float],
#             Tensor
#         ]] = None,
#         loss_names: Optional[Tuple[str, ...]] = None,
#         learning_rate = 1e-4,
#         restoring_force_alpha = 0.,
#         grad_norm_parameters: Optional[Parameter] = None,
#         accelerator: Optional[Accelerator] = None,
#         frozen = False,
#         initial_losses_decay = 1.,
#         update_after_step = 0.,
#         update_every = 1.
#     ):
#         super().__init__()
#         assert exists(num_losses) or exists(loss_weights)

#         if exists(loss_weights):
#             if isinstance(loss_weights, list):
#                 loss_weights = torch.tensor(loss_weights)

#             num_losses = default(num_losses, loss_weights.numel())
#         else:
#             loss_weights = torch.ones((num_losses,), dtype = torch.float32)

#         assert len(loss_weights) == num_losses
#         assert num_losses > 1, 'only makes sense if you have multiple losses'
#         assert loss_weights.ndim == 1, 'loss weights must be 1 dimensional'

#         self.accelerator = accelerator
#         self.num_losses = num_losses
#         self.frozen = frozen

#         self.loss_names = loss_names
#         assert not exists(loss_names) or len(loss_names) == num_losses

#         assert restoring_force_alpha >= 0.

#         self.alpha = restoring_force_alpha
#         self.has_restoring_force = self.alpha > 0

#         self._grad_norm_parameters = [grad_norm_parameters] # hack

#         # loss weights, either learned or static

#         self.register_buffer('loss_weights', loss_weights)

#         self.learning_rate = learning_rate

#         # initial loss
#         # if initial loss decay set to less than 1, will EMA smooth the initial loss

#         assert 0 <= initial_losses_decay <= 1.
#         self.initial_losses_decay = initial_losses_decay

#         self.register_buffer('initial_losses', torch.zeros(num_losses))

#         # for renormalizing loss weights at end

#         self.register_buffer('loss_weights_sum', self.loss_weights.sum())

#         # for gradient accumulation

#         self.register_buffer('loss_weights_grad', torch.zeros_like(loss_weights), persistent = False)

#         # step, for maybe having schedules etc

#         self.register_buffer('step', torch.tensor(0.))

#         # can update less frequently, to save on compute

#         self.update_after_step = update_after_step
#         self.update_every = update_every

#         self.register_buffer('initted', torch.tensor(False))

#     @property
#     def grad_norm_parameters(self):
#         return self._grad_norm_parameters[0]

#     def backward(self, *args, **kwargs):
#         return self.forward(*args, **kwargs)

#     @beartype
#     def forward(
#         self,
#         losses: Union[
#             Dict[str, Tensor],
#             List[Tensor],
#             Tuple[Tensor],
#             Tensor
#         ],
#         activations: Optional[Tensor] = None,     # in the paper, they used the grad norm of penultimate parameters from a backbone layer. but this could also be activations (say shared image being fed to multiple discriminators)
#         freeze = False,                           # can optionally freeze the learnable loss weights on forward
#         scale = 1.,
#         grad_step = True,
#         **backward_kwargs
#     ):
#         # backward functions dependent on whether using hf accelerate or not
#         # import pdb; pdb.set_trace()
#         backward = self.accelerator.backward if exists(self.accelerator) else lambda l, **kwargs: l.backward(**kwargs)
#         backward = partial(backward, **backward_kwargs)

#         # increment step

#         step = self.step.item()

#         self.step.add_(int(self.training and grad_step))

#         # loss can be passed in as a dictionary of Dict[str, Tensor], will be ordered by the `loss_names` passed in on init

#         if isinstance(losses, tuple) and hasattr(losses, '_asdict'):
#             losses = losses._asdict()

#         if isinstance(losses, dict):
#             assert exists(self.loss_names)
#             input_loss_names = set(losses.keys())
#             assert input_loss_names == set(self.loss_names), f'expect losses named {self.loss_names} but received {input_loss_names}'

#             losses = [losses[name] for name in self.loss_names]

#         # validate that all the losses are a single scalar

#         assert all([loss.numel() == 1 for loss in losses])

#         # cast losses to tensor form

#         if isinstance(losses, (list, tuple)):
#             losses = torch.stack(losses)

#         # auto move gradnorm module to the device of the losses

#         if self.initted.device != losses.device:
#             self.to(losses.device)

#         assert losses.ndim == 1, 'losses must be 1 dimensional'
#         assert losses.numel() == self.num_losses, f'you instantiated with {self.num_losses} losses but passed in {losses.numel()} losses'

#         total_weighted_loss = (losses * self.loss_weights.detach()).sum()

#         backward(total_weighted_loss * scale, **{**backward_kwargs, 'retain_graph': not freeze})

#         # handle base frozen case, so one can freeze the weights after a certain number of steps, or just to a/b test against learned gradnorm loss weights

#         if (
#             self.frozen or \
#             freeze or \
#             not self.training or \
#             step < self.update_after_step or \
#             (step % self.update_every) != 0
#         ):
#             return total_weighted_loss

#         # store initial loss

#         if self.has_restoring_force:
#             if not self.initted.item():
#                 initial_losses = maybe_distributed_mean(losses)
#                 self.initial_losses.copy_(initial_losses)
#                 self.initted.copy_(True)

#             elif self.initial_losses_decay < 1.:
#                 meaned_losses = maybe_distributed_mean(losses)
#                 self.initial_losses.lerp_(meaned_losses, 1. - self.initial_losses_decay)

#         # determine which tensor to get grad norm from

#         grad_norm_tensor = default(activations, self.grad_norm_parameters)

#         assert exists(grad_norm_tensor), 'you need to either set `grad_norm_parameters` on init or `activations` on backwards'

#         grad_norm_tensor.requires_grad_()

#         # get grad norm with respect to each loss

#         grad_norms = []
#         loss_weights = self.loss_weights.clone()
#         loss_weights = Parameter(loss_weights)

#         for weight, loss in zip(loss_weights, losses):
#             gradients, = grad(weight * loss, grad_norm_tensor, create_graph = True, retain_graph = True)

#             grad_norm = gradients.norm(p = 2)
#             grad_norms.append(grad_norm)

#         grad_norms = torch.stack(grad_norms)

#         # main algorithm for loss balancing

#         grad_norm_average = maybe_distributed_mean(grad_norms.mean())

#         if self.has_restoring_force:
#             loss_ratio = losses.detach() / self.initial_losses

#             relative_training_rate = l1norm(loss_ratio) * self.num_losses

#             gradient_target = (grad_norm_average * (relative_training_rate ** self.alpha)).detach()
#         else:
#             gradient_target = repeat(grad_norm_average, ' -> l', l = self.num_losses).detach()

#         grad_norm_loss = F.l1_loss(grad_norms, gradient_target)

#         backward(grad_norm_loss * scale)

#         # accumulate gradients

#         self.loss_weights_grad.add_(loss_weights.grad)

#         if not grad_step:
#             return

#         # manually take a single gradient step

#         updated_loss_weights = loss_weights - self.loss_weights_grad * self.learning_rate

#         renormalized_loss_weights = l1norm(updated_loss_weights) * self.loss_weights_sum

#         self.loss_weights.copy_(renormalized_loss_weights)

#         self.loss_weights_grad.zero_()
        
#         # 手动释放计算图
#         torch.cuda.empty_cache()
#         del total_weighted_loss, grad_norm_tensor, grad_norms, loss_weights, gradients, grad_norm_average, gradient_target, grad_norm_loss, updated_loss_weights, renormalized_loss_weights
