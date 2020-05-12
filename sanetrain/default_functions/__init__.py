from . import default_train, default_test, default_display

simple_defaults = {'train_step': 'sanetrain.default_functions.default_train.train_step',
                   'train_epoch': 'sanetrain.default_functions.default_train.train_epoch',
                   'test_step': 'sanetrain.default_functions.default_test.test_step',
                   'test_epoch': 'sanetrain.default_functions.default_test.test_epoch',
                   'display_stats': 'sanetrain.default_functions.default_display.display_stats',
}

__all__ = ['simple_defaults']
