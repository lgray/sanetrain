import argparse
import subprocess
from uuid import uuid1

import yaml
from jinja2 import Environment, PackageLoader

from sanetrain.workflow_builder import generate_training

parser = argparse.ArgumentParser(description='Run SaneTrain.')
parser.add_argument("--file", help="Path to training yaml file")
args = parser.parse_args()

env = Environment(loader=PackageLoader('sanetrain', 'templates'))
template = env.get_template('template_mlflow.py')

with open(args.file) as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)

    train_script = generate_training(template, config)
    fname = 'tests/%s.py' % uuid1().hex

    with open(fname, 'w+') as fout:
        fout.write(train_script)

    subprocess.run(["python", fname])

