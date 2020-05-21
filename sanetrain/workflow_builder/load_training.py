import yaml

from ..default_functions import simple_defaults


def generate_training(template, config, defaults=simple_defaults):
    train_loader_params = config.pop('train_loader_params', {})
    config['train_loader_params'] = ', '.join(['%s=%s'%(k, v) for k, v in train_loader_params.items()])
    test_loader_params = config.pop('test_loader_params', {})
    config['test_loader_params'] = ', '.join(['%s=%s'%(k, v) for k, v in test_loader_params.items()])

    model_spec = config.pop('model_spec')
    config['model_import'] = model_spec['import']
    config['model_name'] = model_spec['name']

    model_params = config.pop('model_params', {})
    config['model_params'] = ', '.join(['%s=%s'%(k, v) for k, v in model_params.items()])

    optimizer_params = config.pop('optimizer_params', {})
    config['optimizer_params'] = ', '.join(['%s=%s'%(k, v) for k, v in optimizer_params.items()])

    training_spec = config.pop('training_spec')
    config['train_epoch'] = training_spec['train_epoch'].pop('definition')
    if config['train_epoch'] == 'default':
        config['train_epoch'] = defaults['train_epoch']
    config['train_kwargs'] = ', '.join(['%s=%s'%(k, v) for k, v in training_spec['train_epoch'].items()])

    config['train_step'] = training_spec['train_step'].pop('definition')
    if config['train_step'] == 'default':
        config['train_step'] = defaults['train_step']

    testing_spec = config.pop('testing_spec')
    config['test_epoch'] = testing_spec['test_epoch'].pop('definition')
    if config['test_epoch'] == 'default':
        config['test_epoch'] = defaults['test_epoch']
    config['test_kwargs'] = ', '.join(['%s=%s'%(k, v) for k, v in testing_spec['test_epoch'].items()])

    config['test_step'] = testing_spec['test_step'].pop('definition')
    if config['test_step'] == 'default':
        config['test_step'] = defaults['test_step']

    display_spec = config.pop('display_spec')
    config['display_stats'] = display_spec['definition']
    if config['display_stats'] == 'default':
        config['display_stats'] = defaults['display_stats']

    print(template.render(config))
    return template.render(config)
