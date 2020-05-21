import subprocess
from uuid import uuid1

import yaml
from jinja2 import Environment, PackageLoader

from sanetrain.workflow_builder import generate_training


def test_generate_training():
    env = Environment(loader=PackageLoader('sanetrain', 'templates'))
    template = env.get_template('test_template.py')

    with open('tests/test_model.yaml') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

        train_script = generate_training(template, config)

        fname = 'tests/%s.py' % uuid1().hex

        with open(fname, 'w+') as fout:
            fout.write(train_script)

        subprocess.run(["python", fname])
