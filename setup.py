#!/usr/bin/env python
from setuptools import setup

setup(name='DD-GAN',
      version='1.0',
      description='Library for domain decomposition predictive gan',
      author='Jon Atli Tomasson and Zef Wolffs',
      packages=['ddgan'],
      package_dir={'ddgan': 'ddgan'},
      package_data={'armageddon_model': ['*.csv', '*.txt']},
      include_package_data=True
      )
