"""Setup script for gristmill."""

from setuptools import setup, find_packages

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

setup(
    name='gristmill',
    version='0.1.0dev',
    description=DESCRIPTION.splitlines()[0],
    long_description=DESCRIPTION,
    url='https://github.com/tschijnmo/gristmill',
    author='Jinmo Zhao and Gustavo E Scuseria',
    author_email='tschijnmotschau@gmail.com',
    license='MIT',
    classifiers=CLASSIFIERS,
    packages=find_packages(),
    package_data={'gristmill': ['templates/*']},
    install_requires=['drudge', 'Jinja2']
)
