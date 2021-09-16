# Multiobjective Optimisation

To run the code (on a dummy dataset located in `../data/historic`), use this command after installing the dependencies and activating the virtual environment:

`python3 optimise.py NSGA2`

This command will run the optimisation process with `NSGA-II` save the output of it to the `results/exp{num}` directory, where `num` is an autoincrementing number. To run the code with other parameters, or with other membership functions, you can use command line options that can be viewed using: 

`python3 optimise.py --help`