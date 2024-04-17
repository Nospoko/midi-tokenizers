from setuptools import setup

setup(
    name="midi_tokenizers",
    py_modules=[
        'midi_tokenizers', 
        'midi_trainable_tokenizers', 
        'midi_quantizers', 
        'artifacts', 
        'scripts', 
        'object_generators',
    ],
    install_requires=[
        "datasets~=2.17.1",
        "fortepyan>=0.2.8",
        "numpy~=1.26.3",
        "pandas~=2.1.4",
        "PyYAML~=6.0.1",
    ],
)