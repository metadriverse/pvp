import sys
from distutils.core import setup

from setuptools import find_packages

assert sys.version_info.major == 3 and sys.version_info.minor >= 6, "python version >= 3.6 is required"

setup(
    name='pvp_iclr_release',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        "yapf==0.30.0",
        'gym',
        'numpy',
        'matplotlib',
        'pandas',
        "yapf==0.30.0",
        "ray==1.0.0",
        "ray[all]==1.0.0",
        #"tensorflow==2.3.1",
        "seaborn",
        #"tensorflow-probability==0.11.1",
        "tabulate",
        #"tensorboardX",
        "wandb"
    ]
)
