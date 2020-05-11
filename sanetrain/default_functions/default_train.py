from ..device import device


def train_epoch(optimizer, model, lossf, train_loader, data_unpacker,
                train_step, epoch=None, device=device, **kwargs):
    '''Default function for training one epoch of data'''
    model.train()
    total_loss = 0.
    mask = kwargs.pop('logits_mask', None)
    for data in train_loader:
        model_args, truth = data_unpacker(data)
        data_mask = None if mask is None else data[mask]
        total_loss += train_step(optimizer, model, lossf,
                                 truth, mask=data_mask, device=device,
                                 **model_args)
    return total_loss


def train_step(optimizer, model, lossf, truth, mask, device=device,  **kwargs):
    '''Default function for one batch of training'''
    step_loss = 0.
    optimizer.zero_grad()
    loss = lossf(model(**kwargs), truth) if mask is None else lossf(model(**kwargs)[mask], truth[mask])
    loss.backward()
    step_loss += loss.item()
    optimizer.step()
    return step_loss
