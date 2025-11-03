import torch
_real_torch_load = torch.load
def _full_unpickle_load(f, *args, **kwargs):
    kwargs['weights_only'] = False
    return _real_torch_load(f, *args, **kwargs)
torch.load = _full_unpickle_load
