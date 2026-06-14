import torch
import numpy as np
import multiprocessing
import warnings
import sys
from typing import Any, List, Tuple, Dict, Optional
import pandas as pd


def one_hot_argmax_with_grad(x, dim=-1):
    """
    Memory-efficient one-hot argmax with straight-through gradient estimator.

    Args:
        x (torch.Tensor): Input tensor
        dim (int): Dimension along which to apply argmax (default: -1)
    """
    # Move target dimension to last position for one_hot
    n = x.ndim
    if dim != -1 and dim != n - 1:
        perm = list(range(n))
        perm.pop(dim)
        perm.append(dim)
        x = x.permute(perm)

    # Get one-hot directly
    index = torch.argmax(x, dim=-1)  # [*, num_classes] -> [*]
    y_hard = torch.nn.functional.one_hot(index, num_classes=x.size(-1))  # [*] -> [*, num_classes]

    # Move back if needed
    if dim != -1 and dim != n - 1:
        inv_perm = list(range(n))
        inv_perm.insert(dim, inv_perm.pop())
        y_hard = y_hard.permute(inv_perm)

    # Get softmax for gradient path
    y_soft = torch.softmax(x, dim=dim)

    return y_hard.detach() - y_soft.detach() + y_soft


def causal_conv(inputs, kernel):
    # Assuming shapes:
    # kernel: n_delays x n_in x n_out
    # inputs: n_batch x n_time x n_in

    n_batch = inputs.size(0)
    n_delays = kernel.size(0)
    n_in = inputs.size(2)
    assert inputs.size(2) == kernel.size(
        1), "inputs shape {} (n_batch x n_time x n_in) \n and kernel shape {} (should be n_delays x n_in x n_out)".format(
        inputs.shape, kernel.shape)

    zz = torch.zeros(size=(n_batch, n_delays - 1, n_in), device=inputs.device)
    inputs_padded = torch.cat([zz, inputs], 1)
    output = torch.conv1d(inputs_padded.permute(0, 2, 1), kernel.permute(2, 1, 0)).permute(0, 2, 1)
    return output


def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


def merge_repetition_and_batch_indices(tensor):
    assert len(tensor.shape) == 4

    n_repetition, n_batch, n_time, n_neurons = tensor.shape
    return tensor.reshape([n_repetition * n_batch, n_time, n_neurons])


def cosine_similarity(a,b, epsilon=1e-8):
    assert len(a.shape) == len(b.shape)
    assert a.shape == b.shape
    a = a.flatten()
    b = b.flatten()
    corr = lambda z1 ,z2 : (z1 * z2).sum()
    normalizer = torch.sqrt(corr(a,a) * corr(b,b)).clamp(min=epsilon)
    return corr / normalizer


def compress_raster_plot(spikes):

    n_batch, n_time, n_neurons = spikes.shape
    raster = np.empty(shape=(n_batch,n_neurons), dtype=List[int])


def change_learning_rate(optim, new_lr):
    old_lr = optim.param_groups[0]['lr']
    if old_lr != new_lr:
        print(f"set learning rate: {old_lr} -> {new_lr}")
        optim.param_groups[0]['lr'] = new_lr


def pearson_correlation(a, b, dim_deviation=0, epsilon=1e-8):
    assert dim_deviation in [0,1]
    if dim_deviation == 1:
        a = a.T
        b = b.T

    assert len(a.shape) == len(b.shape)

    if len(a.shape) == 4: raise NotImplementedError()
    if len(a.shape) == 3: raise NotImplementedError()

    with torch.no_grad():
        corr = lambda z1, z2: ((z1 - z1.mean(0)) * (z2 - z2.mean(0))).mean(0)
        normalizer = (corr(a, a) * corr(b, b))**0.5
        normalizer = normalizer.clamp(min=epsilon)
        accuracy = corr(a, b) / normalizer
        accuracy = torch.mean(accuracy)
    return accuracy


def relative_diff(t1,t2):
    diff = np.abs(t1 - t2) / (np.abs(t1) + np.abs(t2) + 1e-8)
    return diff


def relative_diff_torch(t1,t2):
    diff = torch.sum(torch.abs(t1 - t2)) / (torch.sum(torch.abs(t1) + torch.abs(t2)) + 1e-8)
    return diff


def merge_batch_dimension(tensor : torch.Tensor, strict=True):

    if strict:
        is_equal = tensor[0:1] == tensor
        assert is_equal.shape == tensor.shape, "verify test dimensions"
        assert torch.all(is_equal), "not equal at tensor with diff: {}".format(relative_diff(tensor, tensor[0:1]).max())
        return tensor[0]

    is_equal = torch.isclose(tensor, tensor[0:1])
    if not torch.all(is_equal):
        is_equal_np = to_numpy(is_equal)
        n_aligned = np.sum(is_equal_np)
        n_total = np.size(is_equal_np)
        if (n_total - n_aligned) / n_total > 0.02:
            # ideally this should be fixed in allen_tensor.py with a better alignment
            # but with less than 4/784 miss aligned frames we continue for the moment.
            warnings.warn("only {}/{} frames are well aligned".format(n_aligned, n_total))

    return tensor[0]


def tensor_summary(t):
    return "min: {} mean: {} max: {}".format(t.min(), t.mean(), t.max())


def fun(f, q_in, q_out):
    while True:
        i, x = q_in.get()
        if i is None:
            break
        q_out.put((i, f(x)))


def parmap(f, X, nprocs=multiprocessing.cpu_count()):
    q_in = multiprocessing.Queue(1)
    q_out = multiprocessing.Queue()

    proc = [multiprocessing.Process(target=fun, args=(f, q_in, q_out))
            for _ in range(nprocs)]
    for p in proc:
        p.daemon = True
        p.start()

    sent = [q_in.put((i, x)) for i, x in enumerate(X)]
    [q_in.put((None, None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]

    [p.join() for p in proc]

    return [x for i, x in sorted(res)]


def deep_detach(tensor: Any):

    if isinstance(tensor, list): return [deep_detach(t) for t in tensor]
    if isinstance(tensor, tuple): return tuple(deep_detach(list(tensor)))
    if isinstance(tensor, dict): return dict([(k, deep_detach(v)) for k, v in tensor.items()])

    if isinstance(tensor, bool): return tensor
    if isinstance(tensor, int): return tensor
    if isinstance(tensor, float): return tensor
    if isinstance(tensor, np.ndarray): return tensor
    if np.isscalar(tensor): return tensor

    if isinstance(tensor, torch.Tensor): return tensor.detach()


def as_scalar(x):
    if isinstance(x, torch.Tensor): x = to_numpy(x)
    if isinstance(x, np.ndarray):
        assert np.size(x) == 1
        x = x.item()
    if isinstance(x, list): raise NotImplementedError()
    assert np.isscalar(x), f"this is not a scalar: {x}"
    return x

def shuffle(x):
    if isinstance(x, list):
        return [x[i] for i in np.random.permutation(len(x))]

    if isinstance(x, np.ndarray):
        return x[np.random.permutation(len(x))]

    raise NotImplementedError(f"got type {type(x)} for {x}")


def to_numpy(tensor : Any):
    if tensor is None: return None
    if isinstance(tensor, list): return [to_numpy(t) for t in tensor]
    if isinstance(tensor, tuple): return tuple(to_numpy(list(tensor)))
    if isinstance(tensor, dict): return dict([(k,to_numpy(v)) for k,v in tensor.items()])
    if isinstance(tensor, bool): return tensor
    if isinstance(tensor, int): return tensor
    if isinstance(tensor, float): return tensor
    if isinstance(tensor, np.ndarray): return tensor if tensor.size != 1 else as_scalar(tensor)
    if isinstance(tensor, pd.Index): return tensor.values
    if isinstance(tensor, pd.DataFrame): return tensor.values
    if isinstance(tensor, torch.Tensor): return to_numpy(tensor.detach().cpu().numpy())
    if np.isscalar(tensor): return tensor
    raise NotImplementedError(f"unknown tensor type {type(tensor)}: {tensor}")


def average_values(tensor : Any):
    if tensor is None: return None
    # WARNING: List is special, structure is averaged, but with tuple/dict structure is kept
    if isinstance(tensor, list):
        first_element = tensor[0]
        if isinstance(first_element, dict):
            list_of_dict = tensor
            return_dict = {}
            for d in list_of_dict:
                assert d.keys() == first_element.keys(), f"only works if they all have sane keys: got {first_element.keys()} and {d.keys()}"
            for k in first_element.keys():
                return_dict[k] = average_values([d[k] for d in list_of_dict])
            return return_dict
        return np.mean([average_values(t) for t in tensor])
    if isinstance(tensor, tuple): return tuple([average_values(t) for t in tensor])
    if isinstance(tensor, dict): return dict([(k,average_values(v)) for k,v in tensor.items()])
    if isinstance(tensor, bool): return tensor
    if isinstance(tensor, int): return tensor
    if isinstance(tensor, float): return tensor
    if isinstance(tensor, np.ndarray): return tensor.mean()
    if isinstance(tensor, pd.Index): return tensor.values
    if isinstance(tensor, pd.DataFrame): return tensor.values
    if np.isscalar(tensor): return tensor
    if isinstance(tensor, torch.Tensor): return to_numpy(tensor.mean())
    raise NotImplementedError(f"got type: {type(tensor)}: {tensor}")

def raise_error_if_nan(tensor : torch.Tensor,msg : str =""):
    if torch.any(torch.isnan(tensor)):
        raise ValueError(msg)


def nanmean(v, *args, inplace=False, **kwargs):
    if not inplace:
        v = v.clone()
    is_nan = torch.isnan(v)
    v[is_nan] = 0
    return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)

def squared_sigmoid(x):
    x_ = x.clamp(min=-1, max=1)
    return torch.where(x > 0, 1 - 0.5 * torch.square(x_ - 1.0), 0.5 * torch.square(x_ + 1))

def squared_sigmoid_inverse(probs : torch.Tensor):
    return torch.where(probs > 0.5, 1 - torch.sqrt(2 *(1 - probs)), torch.sqrt(2 * probs) - 1)

def sigmoid_inverse_clipped(probs : torch.Tensor, epsilon : float=1e-6):
        probs = probs.clamp(min=epsilon, max=1 - epsilon)
        return torch.log(probs) - torch.log(1 - probs)

def compute_firing_rate(raster, dt):
    batch_size, n_t, n_neuron = raster.shape
    spike_count = raster.sum((0, 1))
    average_firing_rate = spike_count / (batch_size * n_t * dt)

    return average_firing_rate


def temporal_filter(spikes : torch.Tensor, tau : float, dt : float, filter_shape="box", pad="reflect"):

    if isinstance(spikes, np.ndarray):
        return to_numpy(temporal_filter(torch.tensor(spikes, dtype=torch.float32), tau, dt, filter_shape))

    if len(spikes.shape) == 2:
        return temporal_filter(spikes.unsqueeze(0), tau, dt, filter_shape).squeeze(0)

    n_t = int(tau / dt /2) +1
    n_max = n_t * 2 + 1

    # padding and reshape
    n_batch, T, n_neurons = spikes.shape
    if pad == "zeros":
        pad_left = torch.zeros([n_batch, n_t, n_neurons], dtype=spikes.dtype, device=spikes.device)
        pad_right = torch.zeros([n_batch, n_t, n_neurons], dtype=spikes.dtype, device=spikes.device)
    elif pad == "reflect":
        pad_left = spikes[:,:n_t].flip(1)
        pad_right = spikes[:,-n_t:].flip(1)
    else:
        raise NotImplementedError()

    spikes = torch.cat([pad_left, spikes, pad_right], 1)
    spikes = spikes.permute([0, 2, 1])
    spikes = spikes.reshape([n_batch * n_neurons, 1, -1])

    if filter_shape == "triangular":
        filter = n_t + 1 - torch.abs(torch.arange(n_max) - n_t).float()
        filter = filter.to(spikes.device)
    elif filter_shape == "box":
        filter = torch.ones([n_max], dtype=spikes.dtype, device=spikes.device)
    else:
        raise NotImplementedError("no implementation for filter {}".format(filter_shape))
    assert filter.sum() > 0
    filter = filter / filter.sum() # normalize to sum 1
    filter = filter.reshape([1, 1 , n_t * 2 +1])
    res = torch.conv1d(spikes, weight=filter)
    res = res.reshape([n_batch, n_neurons, T])
    res = res.permute([0, 2, 1])
    return res


def torch_binary_mask(shape, prob):
    return torch.rand(shape) > prob


def torch_mask_before_t(tensor, axis, t):
    shp = tensor.shape
    shape_a = []
    shape_b = []
    for k, d in enumerate(shp):

        if k == axis:
            shape_a.append(t)
            shape_b.append(d - t)
        else:
            shape_a.append(d)
            shape_b.append(d)

    shape_a = tuple(shape_a)
    shape_b = tuple(shape_b)

    a = torch.zeros(shape_a, dtype=torch.bool, device=tensor.device)
    b = torch.ones(shape_b, dtype=torch.bool, device=tensor.device)

    return torch.cat([a, b], dim=axis)


def truncate_raster(raster, neuron_indices):
    first_neuron = neuron_indices[0]
    last_neuron = neuron_indices[-1]
    return raster[:, :, first_neuron:last_neuron + 1]


def roll_left_and_append(tensor, new_element, axis):
    # type: (Tensor,Tensor,int) -> Tensor
    # roll the buffer and append new_element at the end:

    n_axis = len(tensor.shape)

    axis_perm = torch.range(0,n_axis)  # torch.range(start=0,end=n_axis,step=1)
    axis_perm[0] = axis
    axis_perm[axis] = 0

    # perform the concatenation
    tensor = tensor.permute(axis_perm)

    if new_element is None:
        new_element = torch.zeros_like(tensor[0])

    new_buffer = torch.cat([tensor[1:], new_element.unsqueeze(0)], 0)
    new_buffer = new_buffer.permute(axis_perm)
    return new_buffer


def roll_right_and_prepend(tensor, new_element=None, axis=0):
    # roll the buffer and append new_element at the end:

    n_axis = len(tensor.shape)
    axis_perm = list(range(n_axis))  # torch.range(start=0,end=n_axis,step=1)
    axis_perm[0] = axis
    axis_perm[axis] = 0

    # perform the concatenation
    tensor = tensor.permute(axis_perm)

    if new_element is None:
        new_element = torch.zeros_like(tensor[0])

    new_buffer = torch.cat([new_element.unsqueeze(0), tensor[:-1]], 0)
    new_buffer = new_buffer.permute(axis_perm)
    return new_buffer


def get_self_conditioning_mask_and_expanded_raster(opt, target_raster, neuron_indices, epsilon_scheduled_sampling):
    # define constants and tensor to condition the data
    n_batch, n_t, n_neuron_targeted = target_raster.shape
    n_neurons_recorded = len(neuron_indices)
    n_neurons_before = neuron_indices[0]
    n_neuron_total = opt.n_rec_srm
    n_neurons_after = n_neuron_total - n_neurons_before - n_neurons_recorded

    # only before t with do not simulate:
    if opt.scheduled_sampling and epsilon_scheduled_sampling is not None:
        mask_self_conditioning = torch.rand_like(target_raster) > epsilon_scheduled_sampling
    else:
        n_t_conditioning = int(n_t * opt.conditioning_fraction)
        mask_self_conditioning = torch_mask_before_t(target_raster, 1, n_t_conditioning)

    # every neuron that has do data is simulated
    mask_self_conditioning = pad_with_ones(mask_self_conditioning,2, n_neurons_before, n_neurons_after)
    # add zeros so that the raster has the correct size
    padded_target_raster = pad_with_zeros(target_raster,2, n_neurons_before, n_neurons_after)

    return mask_self_conditioning, padded_target_raster


def pad_with_ones(raster, axis, pad_before, pad_after):
    shp = raster.shape
    shape_a = []
    shape_b = []

    for k, d in enumerate(shp):
        if k == axis:
            shape_a.append(pad_before)
            shape_b.append(pad_after)
        else:
            shape_a.append(d)
            shape_b.append(d)

    a = torch.ones(size=shape_a, dtype=raster.dtype, device=raster.device)
    b = torch.ones(size=shape_b, dtype=raster.dtype, device=raster.device)
    return torch.cat([a, raster, b], dim=axis)


def pad_with_zeros(raster, axis, pad_before, pad_after):
    shp = raster.shape
    shape_a = []
    shape_b = []

    for k, d in enumerate(shp):
        if k == axis:
            shape_a.append(pad_before)
            shape_b.append(pad_after)
        else:
            shape_a.append(d)
            shape_b.append(d)

    a = torch.zeros(size=shape_a, dtype=raster.dtype, device=raster.device)
    b = torch.zeros(size=shape_b, dtype=raster.dtype, device=raster.device)
    return torch.cat([a, raster, b], dim=axis)


def find_indices_of_a_in_b(a,b):
    # Returns an array c of size a,
    # where the element c_i it the index of the element in b which are the same value as a_i
    sorter = np.argsort(b)
    return sorter[np.searchsorted(b, a, sorter=sorter)]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    x_line = np.linspace(-3, 3, 50)
    y = squared_sigmoid(torch.tensor(x_line))
    x_ = squared_sigmoid_inverse(y)
    y = to_numpy(y)
    x_ = to_numpy(x_)
    plt.plot(x_line, y)
    plt.plot(x_line, x_)
    plt.show()