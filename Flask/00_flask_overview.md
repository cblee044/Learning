# Overview

## Python Version
Flask supports Python 3.7 and newer.

## Dependencies
These are automaticall installed with Flask
* Werkzeug: Implements WSGI, interface between applications and servers.
* Jinja: Template language that renders pages on applications.
* MarkupSafe: Security.
* ItsDangerous: More security.
* Click: Framework for writing command line applications.

## Virtual Environments
Virtual environments ensure version stability for each created project. A Python update, package update, and installation of two incompatible packages may result in a functional script/program to break. Virtual environments tailor a space for projects to run without any of these worries. 

Python includes the `venv` module for virtual environment creation.

### Create an environment
Create a new repository and a virtual environment folder within it
```
mkdir myproject
cd myproject
python3 -m venv venv
```
### Activate the environment
```
. venv/bin/activate
```
## Install Flask
```
pip install Flask
```