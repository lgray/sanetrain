import argparse
import subprocess
from uuid import uuid1

from sanetrain.workflow_builder import generate_training
from sanetrain.workflow_builder import template_mlflow

parser = argparse.ArgumentParser(description='Run SaneTrain.')
parser.add_argument("--file", help="Path to training yaml file")
args = parser.parse_args()

train_script = generate_training(template_mlflow.main_loop, args.file)
fname = 'tests/%s.py' % uuid1().hex

with open(fname, 'w+') as fout:
    fout.write(train_script)

subprocess.run(["python", fname])

