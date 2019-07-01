from typing import Union, Sequence

from torch.optim import Optimizer


class WeightDecayOptimizerWrapper(Optimizer):
    def __init__(self, optimizer: Optimizer, weight_decay: Union[Sequence[float], float], change_with_lr: bool = True) -> None:
        self.optimizer = optimizer
        if isinstance(weight_decay, (list, tuple)):
            assert len(weight_decay) == len(self.optimizer.param_groups)
            assert all((x >= 0 for x in weight_decay))
            self.weight_decays = weight_decay
        else:
            assert weight_decay >= 0
            self.weight_decays = [weight_decay] * \
                len(self.optimizer.param_groups)
        self.state = self.optimizer.state
        self.change_with_lr = change_with_lr

    def step(self, closure=None) -> None:
        for group, weight_decay in zip(self.optimizer.param_groups, self.weight_decays):
            for param in group['params']:
                if param.grad is None or weight_decay == 0:
                    continue
                if self.change_with_lr:
                    param.data.add_(
                        -weight_decay * group['lr'], param.data)
                else:
                    param.data.add_(-weight_decay, param.data)
        self.optimizer.step()

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def add_param_group(self, param_group):
        self.optimizer.add_param_group(param_group)

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def state_dict(self):
        return self.optimizer.state_dict()

    def __repr__(self):
        return self.optimizer.__repr__()

    def __getstate__(self):
        return self.optimizer.__getstate__()

    def __setstate__(self, state):
        self.optimizer.__setstate__(state)
        self.state = self.optimizer.state

    @property
    def param_groups(self):
        return self.optimizer.param_groups
