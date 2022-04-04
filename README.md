# SoundEditor
 
## Installation
This repository uses **python 3.10** in the developing process, but it should work with older 
Versions (3.5, 3.6, ...) as well, only the requirements_(dev).txt file(s) won't work because
of package versions.

It comes with two requirement files. One for running the application (requirements.txt) and one
for doing testing and PEP8 style checking as well (requirements_dev.txt). Those can be installed
via pip:
```
pip install -r requirements_dev.txt
```

The application itself has to be installed as a pip package as well, in order to run the application
and the tests. For this, run
```
pip install -e .
```
in the project root directory. The flag `-e` can be ignored, but any change in repositry has to be accounted
for by a re-installation of the package.

## How to use this repository?

There are two main parts:
- The rapid prototyping jupyter notebook that contains the bleeding edge features hidden in smelly code (sounddevice.ipynb)
- The partly tested application code in the `application.py` file, which can be used to run the application via it's main function

## Application.py

The application loads at start up the `.wav` file `test.wav` and uses it for further processing.

Pressing the g key starts the Gaussian insertion mode

The rest of this readme is a task for my future me ;)
