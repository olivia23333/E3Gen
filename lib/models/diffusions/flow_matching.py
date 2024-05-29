import torch
import torch.nn as nn
from typing import Union
from copy import deepcopy
import torchdyn
from torchdyn.core import NeuralODE
from mmgen.models.builder import MODULES, build_module


def pad_t_like_x(t, x):
    """Function to reshape the time vector t by the number of dimensions of x.

    Parameters
    ----------
    x : Tensor, shape (bs, *dim)
        represents the source minibatch
    t : FloatTensor, shape (bs)

    Returns
    -------
    t : Tensor, shape (bs, number of x dimensions)

    Example
    -------
    x: Tensor (bs, C, W, H)
    t: Vector (bs)
    pad_t_like_x(t, x): Tensor (bs, 1, 1, 1)
    """
    if isinstance(t, (float, int)):
        return t
    return t.reshape(-1, *([1] * (x.dim() - 1)))


@MODULES.register_module()
class ConditionalFlowMatcher(nn.Module):
    """Base class for conditional flow matching methods. This class implements the independent
    conditional flow matching methods from [1] and serves as a parent class for all other flow
    matching methods.

    It implements:
    - Drawing data from gaussian probability path N(t * x1 + (1 - t) * x0, sigma) function
    - conditional flow matching ut(x1|x0) = x1 - x0
    - score function $\nabla log p_t(x|x0, x1)$
    """

    def __init__(self, unet, sigma: Union[float, int] = 0.0, weight_scale=1., train_cfg=None, 
        test_cfg=None):
        r"""Initialize the ConditionalFlowMatcher class. It requires the hyper-parameter $\sigma$.

        Parameters
        ----------
        sigma : Union[float, int]
        """
        super().__init__()
        self.sigma = sigma
        self.unet = build_module(
            unet,
            default_args=dict(
                num_classes=0, num_timesteps=1000, use_rescale_timesteps=False))
        self.weight_scale = weight_scale

        self.train_cfg = deepcopy(train_cfg) if train_cfg is not None else dict()
        self.test_cfg = deepcopy(test_cfg) if test_cfg is not None else dict()

    def compute_mu_t(self, x0, x1, t):
        """
        Compute the mean of the probability path N(t * x1 + (1 - t) * x0, sigma), see (Eq.14) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)

        Returns
        -------
        mean mu_t: t * x1 + (1 - t) * x0

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        raise NotImplementedError('compute_mu_t not implemented!')
        # t = pad_t_like_x(t, x0)
        # return t * x1 + (1 - t) * x0

    def compute_sigma_t(self, t):
        """
        Compute the standard deviation of the probability path N(t * x1 + (1 - t) * x0, sigma), see (Eq.14) [1].

        Parameters
        ----------
        t : FloatTensor, shape (bs)

        Returns
        -------
        standard deviation sigma

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        raise NotImplementedError('compute_sigma_t not implemented!')
        # del t
        # return self.sigma

    def sample_xt(self, x0, x1, t, epsilon):
        """
        Draw a sample from the probability path N(t * x1 + (1 - t) * x0, sigma), see (Eq.14) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)
        epsilon : Tensor, shape (bs, *dim)
            noise sample from N(0, 1)

        Returns
        -------
        xt : Tensor, shape (bs, *dim)

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        mu_t = self.compute_mu_t(x0, x1, t)
        sigma_t = self.compute_sigma_t(t)
        sigma_t = pad_t_like_x(sigma_t, x0)
        return mu_t + sigma_t * epsilon

    def compute_conditional_flow(self, x0, x1, t, xt):
        """
        Compute the conditional vector field ut(x1|x0) = x1 - x0, see Eq.(15) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn from probability path pt

        Returns
        -------
        ut : conditional vector field ut(x1|x0) = x1 - x0

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        raise NotImplementedError('compute_conditional_flow not implemented!')
        # del t, xt
        # return x1 - x0

    def sample_noise_like(self, x):
        return torch.randn_like(x)

    def sample_location_and_conditional_flow(self, x0, x1, t=None, return_noise=False):
        """
        Compute the sample xt (drawn from N(t * x1 + (1 - t) * x0, sigma))
        and the conditional vector field ut(x1|x0) = x1 - x0, see Eq.(15) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        (optionally) t : Tensor, shape (bs)
            represents the time levels
            if None, drawn from uniform [0,1]
        return_noise : bool
            return the noise sample epsilon


        Returns
        -------
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn from probability path pt
        ut : conditional vector field ut(x1|x0) = x1 - x0
        (optionally) eps: Tensor, shape (bs, *dim) such that xt = mu_t + sigma_t * epsilon

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        if t is None:
            t = torch.rand(x0.shape[0]).type_as(x0)
        assert len(t) == x0.shape[0], "t has to have batch size dimension"

        eps = self.sample_noise_like(x0)
        xt = self.sample_xt(x0, x1, t, eps)
        ut = self.compute_conditional_flow(x0, x1, t, xt)
        if return_noise:
            return t, xt, ut, eps
        else:
            return t, xt, ut

    def forward(self, data, return_loss=True, **kwargs):
        if return_loss:
            x0 = torch.randn_like(data)
            t, xt, ut = self.sample_location_and_conditional_flow(x0, data)
            vt = self.unet(xt, t)
            loss = torch.mean((vt - ut) ** 2) * self.weight_scale
            log_vars = dict()
            log_vars.update(loss_ddpm_mse=float(loss))
            return loss, log_vars
        else:
            gen_sample = self.forward_test(data)
            return gen_sample

    def forward_test(self, noise):
        node_ = NeuralODE(self.unet, solver="midpoint", sensitivity="adjoint")
        with torch.no_grad():
            results = node_.trajectory(
            noise,
            t_span=torch.linspace(0, 1, 100, device=noise.device),
            )
        return results

    def compute_lambda(self, t):
        """Compute the lambda function, see Eq.(23) [3].

        Parameters
        ----------
        t : FloatTensor, shape (bs)

        Returns
        -------
        lambda : score weighting function

        References
        ----------
        [4] Simulation-free Schrodinger bridges via score and flow matching, Preprint, Tong et al.
        """
        sigma_t = self.compute_sigma_t(t)
        return 2 * sigma_t / (self.sigma**2 + 1e-8)


@MODULES.register_module()
class TargetConditionalFlowMatcher(ConditionalFlowMatcher):
    """Lipman et al. 2023 style target OT conditional flow matching. This class inherits the
    ConditionalFlowMatcher and override the compute_mu_t, compute_sigma_t and
    compute_conditional_flow functions in order to compute [2]'s flow matching.

    [2] Flow Matching for Generative Modelling, ICLR, Lipman et al.
    """

    def compute_mu_t(self, x0, x1, t):
        """Compute the mean of the probability path tx1, see (Eq.20) [2].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)

        Returns
        -------
        mean mu_t: t * x1

        References
        ----------
        [2] Flow Matching for Generative Modelling, ICLR, Lipman et al.
        """
        del x0
        t = pad_t_like_x(t, x1)
        return t * x1

    def compute_sigma_t(self, t):
        """
        Compute the standard deviation of the probability path N(t x1, 1 - (1 - sigma) t), see (Eq.20) [2].

        Parameters
        ----------
        t : FloatTensor, shape (bs)

        Returns
        -------
        standard deviation sigma 1 - (1 - sigma) t

        References
        ----------
        [2] Flow Matching for Generative Modelling, ICLR, Lipman et al.
        """
        return 1 - (1 - self.sigma) * t

    def compute_conditional_flow(self, x0, x1, t, xt):
        """
        Compute the conditional vector field ut(x1|x0) = (x1 - (1 - sigma) t)/(1 - (1 - sigma)t), see Eq.(21) [2].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn from probability path pt

        Returns
        -------
        ut : conditional vector field ut(x1|x0) = (x1 - (1 - sigma) t)/(1 - (1 - sigma)t)

        References
        ----------
        [1] Flow Matching for Generative Modelling, ICLR, Lipman et al.
        """
        del x0
        t = pad_t_like_x(t, x1)
        return (x1 - (1 - self.sigma) * xt) / (1 - (1 - self.sigma) * t)
