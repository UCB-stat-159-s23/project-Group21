[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/LiaEl886)
# A Study of a Heart Failure Prediction Dataset

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/UCB-stat-159-s23/project-Group21.git/HEAD)

Jupyter book: https://ucb-stat-159-s23.github.io/project-Group21/main.html

The purpose of this project is to analyze the prediction of Heart failure. Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worldwide. Four out of 5CVD deaths are due to heart attacks and strokes, and one-third of these deaths occur prematurely in people under 70 years of age. Heart failure is a common event caused by CVDs and this dataset contains 11 features that can be used to predict a possible heart disease. People with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidaemia or already established disease) need early detection and management wherein a machine learning model can be of great help.

In this project, we conduct EDA analyses of the dataset, and apply various models to predict heart disease.

## Authors
Group 21: Hannah Cooper, Joselin Hartanto, Nancy Xu, James Xu

## Data
Data sourced from https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction. This dataset was created by combining different datasets already available independently but not combined before. In this dataset, 5 heart datasets are combined over 11 common features which makes it the largest heart disease dataset available so far for research purposes. The five datasets used for its curation are:

- Cleveland: 303 observations
- Hungarian: 294 observations
- Switzerland: 123 observations
- Long Beach VA: 200 observations
- Stalog (Heart) Data Set: 270 observations

Total: 1190 observations, Duplicated: 272 observations, Final dataset: 918 observations

## Analysis
- `main.ipynb` : main narrative notebook that summarizes the results for EDA and modeling
- `eda.ipynb` : notebook and functions for EDA analyses and data cleaning
- `modeling.ipynb` : notebook and functions for modeling to predict heart disease

## Installation
To install the package, follow the command on your terminal:

Create the environment: `make env`

Run all notebooks: `make all`

Build the jupyter book: `make html`

## Testing
To test the `analysis/utils.py` functions, navigate to the root directory and run `pytest`

## License
The project is released under the BSD 3-clause License.
