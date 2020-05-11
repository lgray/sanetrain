import yaml

import ..default_functions as defaults
from . import templates


def generate_training(fname):
    config = None
    with open(fname) as f:
        config = yaml.load_all(f, Loader=yaml.SafeLoader)
    
    epochs = config['epochs']
    
    training_script = templates.main_loop
    
    print(training_script)
    
