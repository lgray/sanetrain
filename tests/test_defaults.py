import yaml
import sanetrain
from sanetrain.workflow_builder import generate_training
from uuid import uuid1

import subprocess
import os

def test_generate_training():
    train_script = generate_training('tests/test_model.yaml')
    fname = 'tests/%s.py' % uuid1().hex

    with open(fname, 'w+') as fout:
        fout.write(train_script)
    
    subprocess.run(["python", fname])
