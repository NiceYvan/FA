import math
import torch

class FastAdaBelief(torch.optim.Optimizer):
    def __init__(self, params, beta_1=0.9, lr=0.01, delta=1e-2, xi_1=0.1,
                 xi_2=0.1, gamma=0.9, weight_decay=1e-2):

        defaults = dict(lr=lr, beta_1=beta_1, delta=delta, xi_1=xi_1,
                        xi_2=xi_2, gamma=gamma, weight_decay=weight_decay)
        super(FastAdaBelief, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(FastAdaBelief, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['hat_g_t'] = torch.zeros_like(
                        p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['v_t'] = torch.zeros_like(
                        p, memory_format=torch.preserve_format)

                lr, delta = group['lr'], group['delta']
                xi_1, xi_2 = group['xi_1'], group['xi_2']
                hat_g_t, v_t = state['hat_g_t'], state['v_t']
                gamma, beta_1 = group['gamma'], group['beta_1']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                time_step = state['step']
                beta_2 = 1 - gamma/time_step

                hat_g_t.mul_(beta_1).add_(grad, alpha=1 - beta_1)
                v_t.mul_(beta_2).addcmul_(grad-hat_g_t, grad-hat_g_t, value=1-beta_2)
                denom = time_step*v_t + delta
                p.addcmul_(hat_g_t, 1/denom, value=-lr)

        return loss
