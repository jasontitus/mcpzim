"""Upstream's optimiser for the quantizer logits.

Upstream trains GSQ's quantizer with Lion and a two-group split - logit parameters at
``training.lr1`` (default 2e-4) with weight decay 1.0, and per-group quantizer scales at
``training.lr2`` (default 1e-4) with no decay (src/trainer.py:70-97, src/config.py:45-61,
configs/local/config.yaml:20-22). The port used Adam at a single lr=0.001 for both.

Lion is not distributable here (``lion_pytorch`` is absent), and the update is short
enough to state exactly, so it is implemented rather than depended on. One step, per
parameter, following the reference algorithm with the sign taken after the interpolation:

    update = grad + weight_decay * param
    param -= lr * sign(beta1 * m + (1 - beta1) * update)
    m = beta2 * m + (1 - beta2) * update

Note that the sign is applied to the blended term, so ``beta1`` weights the *previous*
moment and ``(1 - beta1)`` the current update - the ordering that distinguishes Lion from
Adam and the reason it is implemented here instead of approximated.
"""

import torch


class Lion(torch.optim.Optimizer):
    """Lion with upstream's defaults: betas (0.9, 0.95), weight_decay 1.0."""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.95), weight_decay=0.0):
        if lr <= 0:
            raise ValueError('lr must be positive')
        if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError('betas must be in [0, 1)')
        if weight_decay < 0:
            raise ValueError('weight_decay must be non-negative')
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            weight_decay = group['weight_decay']
            for parameter in group['params']:
                if parameter.grad is None:
                    continue
                update = parameter.grad
                if weight_decay:
                    update = update.add(parameter, alpha=weight_decay)
                state = self.state[parameter]
                if not state:
                    state['exp_avg'] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
                moment = state['exp_avg']
                parameter.add_(moment.mul(beta1).add_(update, alpha=1 - beta1).sign_(), alpha=-lr)
                moment.mul_(beta2).add_(update, alpha=1 - beta2)
        return loss


def quantizer_optimizer(parameters, lr1=2e-4, lr2=1e-4, weight_decay=1.0):
    """Upstream's two-group split: logits at lr1 with decay, scales at lr2 without.

    ``parameters`` is any iterable of named parameters; anything whose name ends in
    ``scales`` goes to the second group, which is how ``CPUOneBit``/``MPSOneBit`` expose
    them (``solver/gsq.py``: ``self.scales`` and ``self.sign_logits``).
    """
    named = list(parameters)
    logits = [p for name, p in named if not name.endswith('scales')]
    scales = [p for name, p in named if name.endswith('scales')]
    if not logits:
        raise ValueError('No quantizer logit parameters to optimise')
    groups = [{'params': logits, 'lr': lr1, 'weight_decay': weight_decay}]
    if scales:
        groups.append({'params': scales, 'lr': lr2, 'weight_decay': 0.0})
    return Lion(groups)
