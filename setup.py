import io
import os
import re

from setuptools import setup

# conditionally add teh documentation commands
try:
    from sphinx.setup_command import BuildDoc
except ImportError:
    cmdclass = {'build_sphinx': BuildDoc}
    cmd_options = {
        'build_sphinx': {
            'project': ('setup.py', "lr-range-test"),
            'version': ('setup.py', "0.0.1"),
            'release': ('setup.py', "0.0.1"),
            'source_dir': ('setup.py', 'docs'),
            'build_dir': ('setup.py', './public')
        }
    }
else:
    cmd_options = {}
    cmdclass = {}


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type(u"")
    with io.open(filename, mode="r", encoding='utf-8') as fd:
        return re.sub(text_type(r':[a-z]+:`~?(.*?)`'), text_type(r'``\1``'), fd.read())


setup(
    name="lr-range-test",
    version="0.0.1",
    license='MIT',

    author="Nichita Utiu",
    author_email="nikita.utiu@gmail.com",

    description="LR range test for pytorch models and/or ignite engines",

    packages=['lr_range_test'],

    cmdclass=cmdclass,
    command_options=cmd_options,

    install_requires=[
        'matplotlib>=3.0.3',
        'numpy>=1.16',
        'pytorch-ignite>=0.2',
        'torch',
        'tqdm>=4.32'
    ],
    extras_require={
        'dev': [
            'Sphinx>=2.0.1',
            'sphinx-autodoc-typehints>=1.6.0'
        ]
    },
    test_suite='tests',

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
