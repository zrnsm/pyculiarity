"""
Pyculiarity
-----------

A Python port of Twitter's AnomalyDetection R Package.

The original R source is available here:
https://github.com/twitter/AnomalyDetection.
"""

from setuptools import setup, find_packages

setup(
    name='pyculiarity',
    version='0.0.1a1',
    description='A sample Python project',
    long_description=__doc__,
    url='https://github.com/nicolasmiller/pyculiarity',
    author='Nicolas Steven Miller',
    author_email='nicolasmiller@gmail.com',
    license='GPL',

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering'
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 2.7',
    ],
    keywords='data anomaly detection pandas timeseries',
    packages=['pyculiarity'],
    install_requires=['numpy', 'scipy', 'pandas', 'pytz',
                      'statsmodels', 'rpy2'],
    extras_require={
        'test': ['nose', 'mock']
    }
)
