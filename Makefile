## Makefile to create and configure environment, and run all notebooks.

.ONESHELL:
SHELL = /bin/bash

## create_environment : Create and configure environment
.PHONY : env
env: 
	source /srv/conda/etc/profile.d/conda.sh
	conda env create -f environment.yml
	conda activate project21
	conda install ipykernel
	python -m ipykernel install --user --name makeProject21 --display-name "IPython - project21"

## html : Build the Jupyterbook
.PHONY : html
html:
	jupyter-book build .

## all : Run all notebooks
.PHONY : all
all :
	jupyter execute main.ipynb
    # jupyter nbconvert --to notebook --execute main.ipynb --