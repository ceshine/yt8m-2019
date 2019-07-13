from functools import wraps

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer


class BaseLRScheduler(_LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):
        """Intentionally not calling super().__init__()"""
        if not isinstance(optimizer, Optimizer):
            flag = False
            try:
                from apex.fp16_utils.fp16_optimizer import FP16_Optimizer
                if isinstance(optimizer, FP16_Optimizer):
                    flag = True
            except ModuleNotFoundError:
                pass
            if not flag:
                raise TypeError('{} is not an Optimizer'.format(
                    type(optimizer).__name__))
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
            last_epoch = 0
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(
            map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.last_epoch = last_epoch

        # New in PyTorch 1.1.0

        # Following https://github.com/pytorch/pytorch/issues/20124
        # We would like to ensure that `lr_scheduler.step()` is called after
        # `optimizer.step()`

        def with_counter(func, opt):
            @wraps(func)
            def wrapper(*args, **kwargs):
                opt._step_count += 1
                return func(*args, **kwargs)
            wrapper._with_counter = True
            return wrapper
        self.optimizer.step = with_counter(self.optimizer.step, self.optimizer)
        self.optimizer._step_count = 0
        self._step_count = 0

        # Start from last_epoch
        self.step(last_epoch)


class TriangularLR(BaseLRScheduler):
    def __init__(self, optimizer, max_mul, ratio, steps_per_cycle, decay=1, last_epoch=-1):
        self.max_mul = max_mul - 1
        self.turning_point = steps_per_cycle // (ratio + 1)
        self.steps_per_cycle = steps_per_cycle
        self.decay = decay
        self.history = []
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        residual = self.last_epoch % self.steps_per_cycle
        multiplier = self.decay ** (self.last_epoch // self.steps_per_cycle)
        if residual <= self.turning_point:
            multiplier *= self.max_mul * (residual / self.turning_point)
        else:
            multiplier *= self.max_mul * (
                (self.steps_per_cycle - residual) /
                (self.steps_per_cycle - self.turning_point))
        new_lr = [
            lr * (1 + multiplier) / (self.max_mul + 1) for lr in self.base_lrs]
        self.history.append(new_lr)
        return new_lr


class GradualWarmupScheduler(BaseLRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Source: https://github.com/ildoonet/pytorch-gradual-warmup-lr/blob/master/warmup_scheduler/scheduler.py
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier <= 1.:
            raise ValueError('multiplier should be greater than 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = self.base_lrs
                    self.finished = True
                return self.after_scheduler.get_lr()
            return self.base_lrs
        return [base_lr / self.multiplier * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if self.finished and self.after_scheduler:
            return self.after_scheduler.step(epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)
