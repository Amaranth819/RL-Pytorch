import torch

'''
    Get torch.device from string
'''
def get_device(device_str):
    assert device_str in ['auto', 'cpu', 'cuda']

    if device_str == 'auto':
        return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    else:
        return torch.device(device_str)


'''
    Cast a np array to tensor
'''
def np_to_tensor(np_array, device = torch.device('cpu')):
    return torch.from_numpy(np_array).to(device)


'''
    Cast a tensor to np array
'''
def tensor_to_np(tensor : torch.Tensor):
    return tensor.detach().cpu().numpy()


'''
    Set learning rate for optimizer
'''
def set_optimizer_lr(optimizer, lr : float):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


'''
    Learning rate decaying
'''
def linear_lr_decay(curr_epoch, total_epoch, init_lr, final_lr):
    return init_lr - curr_epoch / float(total_epoch) * (init_lr - final_lr)


def exponential_lr_decay(curr_epoch, rate, init_lr, final_lr):
    return max(init_lr * rate ** curr_epoch, final_lr)