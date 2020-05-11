from ..device import device


def display_stats(epoch, loss, total, correct, device=device):
    test_acc = correct/total
    print('Epoch: {:02d}, Loss: {:.4f}, Test: {:.4f}'.format(epoch, loss, test_acc))
