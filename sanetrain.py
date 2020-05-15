import subprocess
from uuid import uuid1

from sanetrain.workflow_builder import generate_training

train_script = generate_training('tests/test_model.yaml')
fname = 'tests/%s.py' % uuid1().hex

with open(fname, 'w+') as fout:
    fout.write(train_script)

subprocess.run(["python", fname])

