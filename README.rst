sanetrain - easily reproducible machine learning training
============================================================

.. inclusion-marker-1-do-not-remove

A flexible package for defining repeatable machine learning workflows.

.. inclusion-marker-1-5-do-not-remove

sanetrain is a package that uniformizes the process of creating machine learning
training workflows in a declarative way, so that all ingredients and specificities
of a specific machine learning model can be tracked cleanly in github and interfaces
with many modern training infrastructure frameworks. The main problem that this package
intends to solve is to support easy organization of a codebase into a functioning
training pipeline, and make it so this training pipeline is easily repeatable by others.

sanetrain is a HEP community project collaborating with `iris-hep <http://iris-hep.org/>`_
and is currently a prototype. We welcome input to improve its quality as we progress towards
a first release. Please feel free to contribute at our `github repo
<https://github.com/lgray/sanetrain>`_!

.. inclusion-marker-2-do-not-remove

Installation
============

Install sanetrain like any other Python package:

.. code-block:: bash

    pip install sanetrain

or similar (use ``sudo``, ``--user``, ``virtualenv``, or pip-in-conda if you wish).

Strict dependencies
===================

- `Python <http://docs.python-guide.org/en/latest/starting/installation/>`__ (3.6+)

The following are installed automatically when you install sanetrain with pip:

- `pytorch <https://pytorch.org/get-started/locally/>`__ (1.5.0+);
- and other utility packages, as enumerated in ``setup.py``.

.. inclusion-marker-3-do-not-remove

Documentation
=============
None yet, sorry! We'll work on it!
