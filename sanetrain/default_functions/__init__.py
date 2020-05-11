from . import default_train, default_test, default_display

train_epoch = default_train.train_epoch
train_step = default_train.train_step

test_epoch = default_test.test_epoch
test_step = default_test.test_step

display_stats = default_display.display_stats

__all__ = ['train_epoch', 'train_step', 'test_epoch', 'test_step', 'display_stats']
