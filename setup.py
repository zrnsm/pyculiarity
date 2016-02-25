"""
Usage details and source available here: https://github.com/wdm0006/pyculiarity.

The original R source and examples are available here: https://github.com/twitter/AnomalyDetection.

Copyright and License

Original Python port Copyright 2015 Nicolas Steven Miller

Original R source Copyright 2015 Twitter, Inc and other contributors

Licensed under the GPLv3
"""

from setuptools import setup, find_packages
from codecs import open
from os import path

VERSION = '0.0.2'

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# get the dependencies and installs
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    all_reqs = f.read().split('\n')

install_requires = [x.strip() for x in all_reqs if 'git+' not in x]
dependency_links = [x.strip().replace('git+', '') for x in all_reqs if 'git+' in x]

setup(
    name='pyculiarity',
    version='0.0.2',
    description='A Python port of Twitter\'s AnomalyDetection R Package.',
    long_description=__doc__,
    url='https://github.com/nicolasmiller/pyculiarity',
    author='Nicolas Steven Miller',
    author_email='nicolasmiller@gmail.com',
    license='GPL',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 2.7',
    ],
    keywords='data anomaly detection pandas timeseries',
    packages=['pyculiarity'],
    install_requires=install_requires,
    depedency_links=dependency_links,
    extras_require={
        'test': ['nose', 'mock']
    }
)
