"""Setup script for gristmill."""

import os.path
from setuptools import setup, find_packages, Extension

with open('README.rst', 'r') as readme:
    DESCRIPTION = readme.read()

CLASSIFIERS = [
    'Development Status :: 1 - Planning',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3 :: Only',
    'Topic :: Scientific/Engineering :: Mathematics'
]

PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
INCLUDE_DIRS = [
    os.path.join(PROJ_ROOT, 'deps', i, 'include')
    for i in ['cpypp', 'fbitset', 'libparenth']
]

COMPILE_FLAGS = ['-std=gnu++1z']

parenth = Extension(
    'gristmill._parenth',
    ['gristmill/_parenth.cpp'],
    include_dirs=INCLUDE_DIRS,
    extra_compile_args=COMPILE_FLAGS
)

setup(
    name='gristmill',
    version='0.8.0dev0',
    description=DESCRIPTION.splitlines()[0],
    long_description=DESCRIPTION,
    url='https://github.com/tschijnmo/gristmill',
    author='Jinmo Zhao and Gustavo E Scuseria',
    author_email='tschijnmotschau@gmail.com',
    license='MIT',
    classifiers=CLASSIFIERS,
    packages=find_packages(),
    ext_modules=[parenth],
    package_data={'gristmill': ['templates/*']},
    install_requires=['drudge', 'Jinja2', 'sympy>=1.7', 'numpy', 'networkx>=2.0']
)
