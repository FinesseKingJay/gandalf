from setuptools import setup, find_packages
from codecs import open
from os import path

import scorer


here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='scorer',

    # Versions should comply with PEP440. For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version=scorer.__version__,

    description='Scorer for comaprison of models',
    long_description=long_description,

    url='https://github.com/grammarly/lab/ensemble',
    license='Proprietary',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 4 - Beta',

        'Intended Audience :: Developers',

        'License :: Other/Proprietary License',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],

    keywords='grammarly lib api text nlp development preposition nn',

    packages=find_packages(exclude=['docs', 'tests']),

    install_requires=[
        'grampy',
    ],
)
