[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
# Optimisation of BMP Placements in GHMC Area

Code for optimising placement of Best Management Practices (BMPs) in the Greater Hyderabad Municipal Corporation (GHMC) area, Telangana, India. This repo contains:

1. code for running NSGA-III and C-TAEA multiobjective optimisation algorithms (see folder `multiobjective`) [1].
2. code for running fuzzy optimisation with single objective genetic algorithms and three different membership functions (see folder `fuzzy`).

# Setup

It is recommended to run all code within a Python virtual environment. To create an environment and install dependencies:

1. Create a `python3` environment using the bash commands `virtualenv .venv` or any similar command. 
2. Activate the environment using `source .venv/bin/activate`.
3. Run `pip install -r requirements.txt` to install all the required libraries.

Your environment is now ready to run the code!

# References

[1] Rohit Dwivedula, R. Madhuri, K. Srinivasa Raju, A. Vasan; Multiobjective optimisation and cluster analysis in placement of best management practices in an urban flooding scenario. Water Sci Technol 15 August 2021; 84 (4): 966–984. doi: https://doi.org/10.2166/wst.2021.283
