"""A setuptools based setup module.

See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

import setuptools
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md')) as f:
    long_description = f.read()

setuptools.setup(
    name='imgrvt',
    version='1.0.0',
    author='Alexey Shkarin',
    author_email='alex.shkarin@gmail.com',
    description='Radial variance transform',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/SandoghdarLab/rvt',
    project_urls={
        'Documentation': 'https://github.com/SandoghdarLab/rvt',
        'Source': 'https://github.com/SandoghdarLab/rvt',
        'Tracker': 'https://github.com/SandoghdarLab/rvt/issues'
    },
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Image Processing',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3'
    ],
    packages=['imgrvt'],
    install_requires=['numpy','scipy']
)