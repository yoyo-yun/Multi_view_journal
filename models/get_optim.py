import torch.optim as Optim


class WarmupOptimizer(object):
    def __init__(self, lr_base, optimizer, data_size, batch_size):
        self.optimizer = optimizer
        self._step = 0
        self.lr_base = lr_base
        self._rate = 0
        self.data_size = data_size
        self.batch_size = batch_size


    def step(self):
        self._step += 1

        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate

        self.optimizer.step()


    def zero_grad(self):
        self.optimizer.zero_grad()


    def rate(self, step=None):
        if step is None:
            step = self._step

        if step <= int(self.data_size / self.batch_size * 0.5):
            r = self.lr_base * 1/5.
        elif step <= int(self.data_size / self.batch_size * 1):
            r = self.lr_base * 2/5.
        elif step <= int(self.data_size / self.batch_size * 1.5):
            r = self.lr_base * 3/5.
        elif step <= int(self.data_size / self.batch_size * 2):
            r = self.lr_base * 4/5.
        elif step <= int(self.data_size / self.batch_size * 2.5):
            r = self.lr_base * 5/5.
        elif step <= int(self.data_size / self.batch_size * 3.0):
            r = self.lr_base * 4/5.
        elif step <= int(self.data_size / self.batch_size * 3.5):
            r = self.lr_base * 3/5.
        elif step <= int(self.data_size / self.batch_size * 4):
            r = self.lr_base * 3/5.
        else:
            r = self.lr_base * 2/5.

        return r


def get_optim(__C, model, data_size, lr_base=None):
    if lr_base is None:
        lr_base = __C.TRAIN.lr_base

    return WarmupOptimizer(
        lr_base,
        Optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=0,
            betas=__C.TRAIN.opt_betas,
            eps=__C.TRAIN.opt_eps
        ),
        data_size,
        __C.TRAIN.batch_size
    )


def get_Adam_optim(__C, model, encoder, train_bert=False):
    params_group = [{'params': filter(lambda p: p.requires_grad, model.parameters())}]

    print("===setting additional sequential encoder in trainable...")
    try:
        params_group.append({'params': filter(lambda p: p.requires_grad, encoder.contextualized_encoder.parameters())})
        print("Done!")
    except:
        print("There's no additional sequential encoder!")
    print()

    if train_bert is True:
        params_group.append({'params': filter(lambda p: p.requires_grad, encoder.encoder.parameters())})

    return Optim.Adam(
        params_group,
        lr=__C.TRAIN.lr_base,
        betas=__C.TRAIN.opt_betas,
        eps=__C.TRAIN.opt_eps
    )


def adjust_lr(optim, decay_r):
    optim.lr_base *= decay_r
