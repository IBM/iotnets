import torch


def state_dict_rm_module(state_dict, verbose=True):
    for _ in range(len(state_dict)):
        key, v = state_dict.popitem(False)
        module_list = key.split('.')
        key_new = '.'.join(module_list[1:len(module_list)])
        if verbose:
            print("Replace {} with {}".format(key, key_new))
        state_dict[key_new] = v

    return state_dict


def state_dict_add_module(state_dict, verbose=True):
    for _ in range(len(state_dict)):
        key, v = state_dict.popitem(False)
        key_new = "module.{}".format(key)
        if verbose:
            print("Replace {} with {}".format(key, key_new))
        state_dict[key_new] = v

    return state_dict


def load_ref_ckpt(file, device=None):
    # Load checkpoint.
    print('==> Resuming from checkpoint {} ... '.format(file), end="")
    checkpoint = torch.load(file, map_location=device)
    print("[OK]")
    return checkpoint


def load_net(net, file, device="cuda", mode="None", verbose=True):
    checkpoint = load_ref_ckpt(file, device)
    # print(checkpoint.keys())
    # print(checkpoint)
    state = checkpoint['net']

    if mode == "rm":
        state = state_dict_rm_module(state, verbose)
    elif mode == "add":
        state = state_dict_add_module(state, verbose)
    else:
        pass

    net.load_state_dict(state)
    acc = checkpoint['acc']
    epoch = checkpoint['epoch']
    return net, acc