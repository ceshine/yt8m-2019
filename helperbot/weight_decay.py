from torch.optim import Optimizer


class WeightDecayOptimizerWrapper(Optimizer):
    def __init__(self, optimizer: Optimizer, weight_decay: float) -> None:
        assert weight_decay > 0
        self.optimizer = optimizer
        self.weight_decay = weight_decay

    def step(self, closure=None) -> None:
        for group in self.optimizer.param_groups:
            for param in group['params']:
                param.data = param.data.add(
                    -self.weight_decay * group['lr'], param.data)
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

    @property
    def param_groups(self):
        return self.optimizer.param_groups
