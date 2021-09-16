# Fuzzy Optimisation

To run the code (on a dummy dataset located in`../data/historic`), use this command after installing the dependencies and activating the virtual environment:

`python3 fuzzy_optimise.py GA`

This command will run the optimisation process with `nonlinear` membership function and save the output of it to the `results/` directory. To run the code with other parameters, or with other membership functions, you can use command line options that can be viewed using: 

`python3 fuzzy_optimise.py --help`


Other information:

- `final_results`, containing the results presented in the paper in both .csv and graphical formats. This folder contains three subfolders: `exp0` (results for **hyperbolic** membership function), `exp1` (**exponential membership** function) and `exp2` (**nonlinear membership function**).
- The file `optimisation_progress.csv` provides results on the convergence of the optimisation approaches with each set of parameters.
- Files `hyperbolic.csv`, `exponential.csv` and `nonlinear.csv` contain the final results obtained at the end of optimisation. 
