import numpy as np

from torch.optim import Optimizer


class Stepper(object):
    """ Used to \"step\" from start,end (`vals`) over `n_iter` iterations on a schedule defined by `func` """

    def __init__(self, func, start, end=0, n=1):
        self.start, self.end = start, end
        self.n = max(1, n)
        self.func = func
        self.i = 0
        self.val = start

    def step(self):
        """ Return next value along annealed schedule """
        self.i += 1
        self.val = self.func(self.start, self.end, self.i / self.n)

    def is_done(self):
        """ Return `True` if schedule completed """
        return self.i >= self.n


class OneCycleScheduler(object):
    """ Once cycle learning policy basic implementation """

    @staticmethod
    def annealing_cos(start, end, pct):
        """ Cosine anneal from as pct goes from 0.0 to 1.0 """
        if start < end:
            cos_out = 1 - np.cos(np.pi * pct)
            return start + (end - start) * (cos_out / 2)
        else:
            cos_out = 1 + np.cos(np.pi * pct)
            return end - (end - start) * (cos_out / 2)

    def __init__(self, optimizer, lr_max, batches, epochs, pct_start=0.3, debug=False):
        self.debug = debug
        self.optimizer = optimizer
        self.lr_max = lr_max
        self.div_factor = 25.
        self.lr_low = self.lr_max / self.div_factor
        self.moms = (0.95, 0.85)

        total_steps = batches * epochs
        up_steps = int(total_steps * pct_start)
        down_steps = total_steps - up_steps
        self.lr_phases = (Stepper(self.annealing_cos, self.lr_low, self.lr_max, n=up_steps),
                          Stepper(self.annealing_cos, self.lr_max, self.lr_low / 1e4, n=down_steps))
        self.mom_phases = (Stepper(self.annealing_cos, self.moms[0], self.moms[1], n=up_steps),
                           Stepper(self.annealing_cos, self.moms[1], self.moms[0], n=down_steps))
        self.phase_i = 0
        self.update_lr_mom()

    def update_lr_mom(self):
        lr_stepper = self.lr_phases[self.phase_i]
        mom_stepper = self.mom_phases[self.phase_i]
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr_stepper.val
            param_group['betas'] = (mom_stepper.val, param_group['betas'][1])
        if self.debug:
            print(f'phase_i={self.phase_i}, step={lr_stepper.i}, lr={lr_stepper.val}, mom={mom_stepper.val}')

    def batch_step(self):
        """ Call after batch processed """
        assert self.phase_i < len(self.lr_phases)

        self.lr_phases[self.phase_i].step()
        self.mom_phases[self.phase_i].step()
        self.update_lr_mom()
        if self.lr_phases[self.phase_i].is_done():
            self.phase_i += 1


class CyclicLR(object):
    """ Mostly based on https://github.com/Randl/MobileNetV2-pytorch/blob/master/clr.py """

    def __init__(self, optimizer, base_lr=1e-5, max_lr=1e-2,
                 step_size=150, mode='triangular2', gamma=1.,
                 scale_fn=None, scale_mode='cycle', i=-1, debug=False):
        self.optimizer = optimizer
        self.base_lrs = [base_lr] * len(self.optimizer.param_groups)
        self.max_lrs = [max_lr] * len(self.optimizer.param_groups)
        self.step_size = step_size
        self.debug = debug

        if mode not in ['triangular', 'triangular2', 'exp_range'] \
                and scale_fn is None:
            raise ValueError('mode is invalid and scale_fn is None')

        self.mode = mode
        self.gamma = gamma

        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = self._triangular_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = self._triangular2_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = self._exp_range_scale_fn
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode

        self.i = i
        self.batch_step()

    def batch_step(self):
        self.i += 1
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

        if self.debug:
            print(f'i={self.i}, lr={self.get_lr()}')

    def _triangular_scale_fn(self, x):
        return 1.

    def _triangular2_scale_fn(self, x):
        return 1 / (2. ** (x - 1))

    def _exp_range_scale_fn(self, x):
        return self.gamma ** (x)

    def get_lr(self):
        step_size = float(self.step_size)
        cycle_i = np.floor(1 + self.i / (2 * step_size))
        x = np.abs(self.i / step_size - 2 * cycle_i + 1)

        lrs = []
        for base_lr, max_lr in zip(self.base_lrs, self.max_lrs):
            base_height = (max_lr - base_lr) * np.maximum(0, (1 - x))
            if self.scale_mode == 'cycle':
                lr = base_lr + base_height * self.scale_fn(cycle_i)
            else:
                lr = base_lr + base_height * self.scale_fn(self.i)
            lrs.append(lr)
        return lrs

# CyclicLR test

# class OptimizerMock(object):
#     param_groups = [{'lr': 0.1}]

# cyclic = CyclicLR(OptimizerMock(), base_lr=1e-4, max_lr=1e-2, step_size=155, mode='triangular2')

# def batch_get_lr(c):
#     c.batch_step()
#     return c.get_lr()

# iterations = np.arange(0, 103 * 15)
# lr = [batch_get_lr(cyclic) for i in iterations]
# plt.plot(iterations, lr)


class WarmRestartsLR(object):
    """ Basic implementation of https://arxiv.org/pdf/1608.03983.pdf """

    def __init__(self, optimizer,
                 lr_min=1e-9, lr_max=1e-1, T=50, T_mult=2, cycles=5,
                 i=-1, debug=False):
        self.optimizer = optimizer
        self.debug = debug
        self.T = T
        self.T_mult = T_mult
        self.lr_min, self.lr_max = lr_min, lr_max
        self.cycles = cycles
        self.scale = 1

        self.i = i
        self.batch_step()

    def batch_step(self):
        if self.cycles > 0:
            self.i += 1

            if self.i == self.T:
                self.i = 0
                self.T *= self.T_mult
                self.lr_max *= self.scale
                self.cycles -= 1

            for param_group, lr in zip(self.optimizer.param_groups, [self.get_lr()]):
                param_group['lr'] = lr

        if self.debug:
            print(f'i={self.i}, lr={self.get_lr()}')

    def get_lr(self):
        cos_out = (1 + np.cos(self.i / self.T * np.pi)) / 2
        lr = self.lr_min + (self.lr_max - self.lr_min) * cos_out
        return lr

# WarmRestartsLR test
# class OptimizerMock(object):
#     param_groups = [{'lr': 0.1}]
#
# warm_restarts = WarmRestartsLR(OptimizerMock(), lr_min=1e-4, lr_max=1e-1, T=50, T_mult=2)
#
# def batch_step_lr(sch):
#     sch.batch_step()
#     return sch.get_lr()
#
# iterations = np.arange(0, 103 * 15)
# lr = [batch_step_lr(warm_restarts) for i in iterations]
# fig, ax = plt.subplots()
# ax.set_yscale('log')
# ax.plot(iterations, lr);