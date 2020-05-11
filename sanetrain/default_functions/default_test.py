from ..device import device


def test_epoch(model, lossf, test_loader, data_unpacker,
               epoch=None, device=device, **kwargs):
    '''Default function for testing one epoch of data'''
    model.eval()
    total = 0.
    total_correct = 0.
    total_loss = 0.
    mask = kwargs.pop('logits_mask', None)
    for data in test_loader:
        model_args, truth = data_unpacker(data)
        size, correct, loss = test_step(model, lossf, truth,
                                        mask=mask, device=device,
                                        **model_args)
        total += size
        total_correct += correct
        total_loss += loss
    return {'epoch': epoch,
            'loss': total_loss,
            'total': total,
            'correct': total_correct}


def test_step(model, lossf, truth, mask, device=device, **kwargs):
    '''Default function for one batch of testing'''
    logits = model(**kwargs) if mask is None else model(**kwargs)[mask]
    truth = truth if mask is None else truth[mask]
    pred = logits.max(1)[1]
    correct = pred.eq(truth).sum().item()
    step_loss = lossf(logits, truth).item()
    return len(truth), correct, step_loss
