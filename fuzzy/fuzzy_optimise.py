import argparse
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import os
import time
import pickle
from itertools import combinations
import matplotlib.pyplot as plt
import gc
from itertools import product

# PyMoo Imports
from pymoo.optimize import minimize
from pymoo.factory import get_sampling
from pymoo.factory import get_termination
from pymoo.factory import get_reference_directions
from pymoo.factory import get_decision_making
from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.ga import GA

parser = argparse.ArgumentParser(description='Run an optimization algorithm on the GHMC BMP data.')

# basic arguments
parser.add_argument('model', help='Name of the model to run it on. Must be GA')
parser.add_argument('-d', '--data', type=str, default='historic', help='Run optimisation on "historic" data. Must be "historic"')
parser.add_argument('-p', '--population', type=int, default=100, help='Number of members in the population of the EA')
parser.add_argument('-g', '--generations', type=int, default=-1, help='Number of generations to run the EA. If this is less than zero, default termination used')
parser.add_argument('-s', '--seed', type=int, default=1, help='Seed for reproducibility')


# optimization algorithms and membership functions
parser.add_argument('-m', '--membership', type=str, default='nonlinear', help='Membership function: nonlinear, exponential, hyperbolic.')

args = parser.parse_args()

runtime_information = dict()
for arg in vars(args):
    runtime_information[arg] = getattr(args, arg)

print("Begin running with configuration:")
print(runtime_information)

plot_names = dict()
plot_names["GA"] = "Single Objective Genetic Algorithm (SOGA)"

if args.model not in ["GA"]:
    print("Optimization Model", args.model, "is not defined.")
    exit(-1)

if args.data not in ["historic"]:
    print("Dataset", args.data, "does not exist.")
    exit(-1)

runtimes = dict()

if args.data == "historic":
	# Read nonrooftop data
	start_time = time.time()
	nonrooftop = gpd.read_file("../data/historic/historic_nonrooftop.shp")
	nonrooftop["Runredvol"] /= 1000
	nonrooftop["cost_mult"] = nonrooftop["Depth"] * 3500
	nonrooftop["runred_mult"] = (nonrooftop["Runredeff"] * nonrooftop["Rainfall"])/1000
	nonrooftop["pollutant_mult"] = nonrooftop["pollutant_"] * nonrooftop["peff"] * nonrooftop["runoff"]
	nonrooftop = nonrooftop[
        ["BMP", "Cost", "Runredvol", "Pollutant", "Shape_Area", 
        "cost_mult", "runred_mult", "pollutant_mult", "geometry"]
    ]
	end_time = time.time()
	print("Nonrooftop BMPs:", nonrooftop.shape)
	print("Took %.2f s."%(end_time - start_time))
	runtimes["read_nonrooftop"] = end_time - start_time

	# Read rooftop data
	start_time = time.time()
	rooftop = gpd.read_file("../data/historic/historic_rooftop.shp")
	rooftop["cost_mult"] = ((6 * 3010)/100)
	rooftop["runred_mult"] = rooftop["Rainfall"] * (0.7)/1000
	rooftop["pollutant_mult"] = rooftop["Runoff"] * rooftop["peff"] * 14
	rooftop = rooftop[
        ["BMP", "Cost", "Runredvol", "Pollutant", "Shape_Area", 
         "cost_mult", "runred_mult", "pollutant_mult", "geometry"]
    ]
	end_time = time.time()
	print("Read rooftop. Took %.2f s."%(end_time - start_time))
	runtimes["read_rooftop"] = end_time - start_time

else:
    print("Sample dataset available only for historic event")
    exit(-1)

# bucketing data
bucketed_data = nonrooftop[["BMP", "cost_mult", "runred_mult", "pollutant_mult", "Cost", "Runredvol", "Pollutant", "Shape_Area"]].groupby(["BMP", "cost_mult", "runred_mult", "pollutant_mult"]).agg('sum').reset_index()
bucketed_data = bucketed_data.sort_values(by = "BMP").reset_index().drop("index", axis=1)
print(bucketed_data["BMP"].value_counts())
bucketed_rooftop = rooftop[["BMP", "cost_mult", "runred_mult", "pollutant_mult", "Cost", "Runredvol", "Pollutant", "Shape_Area"]].groupby(["BMP", "cost_mult", "runred_mult", "pollutant_mult"]).agg('sum').reset_index()
bucketed_rooftop = bucketed_rooftop.sort_values(by = "BMP").reset_index().drop("index", axis=1)
print(bucketed_rooftop["BMP"].value_counts())

# calculate constraints
start_time = time.time()
areas = bucketed_data[["BMP"]].drop_duplicates().reset_index(drop=True)
areas["geometry"] = None
areas["plain_sum"] = 0

start_time = time.time()
bioretention = []
vegetated_strip = [] 
grassed_swales = []
sand_filter = []
porous_pavement = []
wet_pond = []
constructed_wetland = []

for i in range(nonrooftop.shape[0]):
    if nonrooftop["BMP"][i] == "Bioretention":
        bioretention.append(nonrooftop["geometry"][i])
        areas.loc[areas["BMP"] == "Bioretention", "plain_sum"] += nonrooftop["geometry"][i].area

    elif nonrooftop["BMP"][i] == "Vegetated_filterstrip":
        vegetated_strip.append(nonrooftop["geometry"][i])
        areas.loc[areas["BMP"] == "Vegetated_filterstrip", "plain_sum"] += nonrooftop["geometry"][i].area

    elif nonrooftop["BMP"][i] == "Grassed_swales":
        grassed_swales.append(nonrooftop["geometry"][i])
        areas.loc[areas["BMP"] == "Grassed_swales", "plain_sum"] += nonrooftop["geometry"][i].area

    elif nonrooftop["BMP"][i] == "Sand_filter__surface_":
        sand_filter.append(nonrooftop["geometry"][i])
        areas.loc[areas["BMP"] == "Sand_filter__surface_", "plain_sum"] += nonrooftop["geometry"][i].area

    elif nonrooftop["BMP"][i] == "Porous_Pavement":
        porous_pavement.append(nonrooftop["geometry"][i])
        areas.loc[areas["BMP"] == "Porous_Pavement", "plain_sum"] += nonrooftop["geometry"][i].area

    elif nonrooftop["BMP"][i] == "Wet_pond":
        wet_pond.append(nonrooftop["geometry"][i])
        areas.loc[areas["BMP"] == "Wet_pond", "plain_sum"] += nonrooftop["geometry"][i].area

    elif nonrooftop["BMP"][i] == "Constructed_wetland":
        constructed_wetland.append(nonrooftop["geometry"][i])
        areas.loc[areas["BMP"] == "Constructed_wetland", "plain_sum"] += nonrooftop["geometry"][i].area

    else: 
        print("Fatal Error: Unrecognized BMP type.")

areas = areas.sort_values(by="BMP")
areas.at[0, "geometry"] = shapely.ops.unary_union(bioretention)
areas.at[1, "geometry"] = shapely.ops.unary_union(constructed_wetland)
areas.at[2, "geometry"] = shapely.ops.unary_union(grassed_swales)
areas.at[3, "geometry"] = shapely.ops.unary_union(porous_pavement)
areas.at[4, "geometry"] = shapely.ops.unary_union(sand_filter)
areas.at[5, "geometry"] = shapely.ops.unary_union(vegetated_strip)   
areas.at[6, "geometry"] = shapely.ops.unary_union(wet_pond)
print("Created all Multipolygons")
print("Took %.2f s."%(time.time() - start_time))

upper_limits = np.ones(shape=(bucketed_data.shape[0] + bucketed_rooftop.shape[0]))
lower_limits = np.zeros(shape=(bucketed_data.shape[0] + bucketed_rooftop.shape[0]))
all_combinations = []
all_constraint_values = []
for i in range(2, 8):
    all_combinations.extend(list(combinations(np.asarray(areas["BMP"]), i)))
for c in all_combinations:
    all_constraint_values.append(shapely.ops.unary_union(np.asarray(areas.loc[areas["BMP"].isin(c)]["geometry"])).area)
end_time = time.time()
print("Evaluated all constraints")
print("Took %.2f s."%(end_time - start_time))
runtimes["constraint_evaluation"] = end_time - start_time

def nonlinear(z, zl, zu, beta):
    tmp1 = (z - zl)/(zu-zl)
    membership = tmp1 ** beta
    return membership

def hyperbolic(z, zl, zu):
    alpha_p = 6.00/(zu - zl)
    z_avg = (zu+zl)/2.00
    tmp1 = (z - z_avg)*alpha_p
    membership = 0.5 * np.tanh(tmp1) + 0.5
    return membership

def exponential(z, zl, zu, s):
    denominator = 1 - np.exp(-1 * s)
    psi = (zu - z)/(zu - zl)
    numerators = np.exp(-1 * s * psi) - np.exp(-1 * s)
    membership = numerators/denominator
    return membership

'''
    Define upper and lower limits of objectives. These will be 
    calculated automatically for the dataset. In the paper, the
    values of these for the entire dataset were:

        ZL_COST = -34972246294.154465 INR
        ZU_RUNRED = 15468781.81909658 m^3
        ZU_POLLUTANT = 110930541997.94527 mg
''' 
ZL_COST = -1 * (np.sum(nonrooftop["Cost"]) + np.sum(rooftop["Cost"])) # in Indian rupees (INR).
ZU_COST = 0

ZL_RUNRED = 0
ZU_RUNRED = np.sum(nonrooftop["Runredvol"]) + np.sum(rooftop["Runredvol"])

ZL_POLLUTANT = 0
ZU_POLLUTANT = np.sum(nonrooftop["Pollutant"]) + np.sum(rooftop["Pollutant"]) 


rooftop = None
nonrooftop = None

# define BMP problem
class BMPProblem(Problem):
    def __init__(self, 
                membership_function = "exponential", 
                beta = [1, 1, 1], 
                s = [1, 1, 1],
                pollutant_lower_bound = 25,
                runred_lower_bound = 35
        ):
        '''
            1. pollutant_lower_bound: refers to the minimum amount of pollutant that must be removed by
                                      a BMP configuration (in mg). In the paper, we used this value to be
                                      pollutant_lower_bound = 25000000000 mg (or 25 tonnes)

            2. runred_lower_bound:    refers to the minimum amount of runoff that must be reduced by the 
                                      BMP configuration (in m^3). In the paper, we used this value to be
                                      runred_lower_bound = 3500000 m^3.

        '''
        super().__init__(n_var=bucketed_data.shape[0] + bucketed_rooftop.shape[0],
                         n_obj=1,
                         n_constr = 122,
                         xl=lower_limits,
                         xu=upper_limits)
        self.membership_function = membership_function
        self.beta = beta
        self.s = s # should be between 0 and 1
        self.pollutant_lower_bound = pollutant_lower_bound
        self.runred_lower_bound = runred_lower_bound
        print("P", self.pollutant_lower_bound)
        print("R", self.runred_lower_bound)

    def _evaluate(self, X, out, *args, **kwargs):
        '''
            X is of size (n, bucketed_data.shape[0] + bucketed_rooftop.shape[0]) since we 
            have that many decision variables.
            
            Other points:

            - all objective functions are represented in *maximise* form.
            - f1 is runoff reduction
            - f2 is pollutant load reduction
            - f3 is cost (Note: we want to minimise cost, but to convert
              it to maximisation form, we multiply it by -1)
        '''
        
        X_nonroof = X[:,:bucketed_data.shape[0]]
        X_roof = X[:, bucketed_data.shape[0]:]
        
        f1_nonroof = np.sum(X_nonroof * np.asarray(bucketed_data["Runredvol"]), axis=1)
        f2_nonroof = np.sum(X_nonroof * np.asarray(bucketed_data["Pollutant"]), axis=1)
        f3_nonroof = -1 * np.sum(X_nonroof * np.asarray(bucketed_data["Cost"]), axis=1)
        
        f1_roof = np.sum(X_roof * np.asarray(bucketed_rooftop["Runredvol"]), axis=1)
        f2_roof = np.sum(X_roof * np.asarray(bucketed_rooftop["Pollutant"]), axis=1)
        f3_roof = -1 * np.sum(X_roof * np.asarray(bucketed_rooftop["Cost"]), axis=1)
        
        f1 = f1_nonroof + f1_roof
        f2 = f2_nonroof + f2_roof
        f3 = f3_nonroof + f3_roof
        
        if self.membership_function == "nonlinear":
            f1_m = nonlinear(f1, ZL_RUNRED, ZU_RUNRED, self.beta[0])
            f2_m = nonlinear(f2, ZL_POLLUTANT, ZU_POLLUTANT, self.beta[1])
            f3_m = nonlinear(f3, ZL_COST, ZU_COST, self.beta[2])
        
        elif self.membership_function == "hyperbolic":
            f1_m = hyperbolic(f1, ZL_RUNRED, ZU_RUNRED)
            f2_m = hyperbolic(f2, ZL_POLLUTANT, ZU_POLLUTANT)
            f3_m = hyperbolic(f3, ZL_COST, ZU_COST)
        
        elif self.membership_function == "exponential":
            f1_m = exponential(f1, ZL_RUNRED, ZU_RUNRED, self.s[0])
            f2_m = exponential(f2, ZL_POLLUTANT, ZU_POLLUTANT, self.s[1])
            f3_m = exponential(f3, ZL_COST, ZU_COST, self.s[2])
        
        else:
            print("Unknown membership function", self.membership_function)
            exit(-1)

        lambda_val = np.column_stack([f1_m, f2_m, f3_m])
        out["F"] = -1 * np.min(lambda_val, axis = 1) 
        
        constraints = []
        
        for i in range(0, len(all_combinations)):
            c = all_combinations[i]
            upper_limit = all_constraint_values[i]            
            subset = bucketed_data.loc[bucketed_data["BMP"].isin(c)]
            ans = np.sum(np.asarray(subset["Shape_Area"]) * X[:, subset.index], axis = 1) - upper_limit
            constraints.append(ans)

        c1 = self.runred_lower_bound - f1
        c2 = self.pollutant_lower_bound - f2
        constraints.append(c1)
        constraints.append(c2)
        out["G"] = np.column_stack(constraints)

# perform optimization
if args.membership == "hyperbolic":
    parameter_options = [-1]
elif args.membership == "exponential":
    parameter_options = [0.2, 0.4, 0.6, 0.8, 1]
elif args.membership == "nonlinear":
    parameter_options = [0.1, 0.4, 1, 3, 5]
else:
    print("Not supported membership function", args.membership)
    exit(-1)

trialParams = list(product(parameter_options, repeat=3))
numTrials = len(trialParams)
print("Performing a total of", numTrials, "using parameter_options", parameter_options)

# for saving results
if not os.path.isdir("results/"):
    os.mkdir("results")

previous_runs = list(filter(lambda x: os.path.isdir("results/"+x), os.listdir("results")))

try:
    curr_ind = max([int(x[3:]) for x in previous_runs]) + 1
except ValueError:
    # if no directories exist in `results`, start numbering from zero.
    curr_ind = 0

BASE_DIRECTORY = "results/exp{ind}/".format(ind=curr_ind)
print("BASE_DIRECTORY:", BASE_DIRECTORY)

os.mkdir(BASE_DIRECTORY)

if args.membership == "hyperbolic":
    physical_interpretation = pd.DataFrame(
        columns = [
            'lambda', 'Runred', 'Cost', 'Pollutant', 
            'Infiltration trench', 'Vegetated_filterstrip', 'Wet_pond', 
            'Bioretention', 'Constructed_wetland', 'Porous_Pavement', 
            'Grassed_swales', 'Sand_filter__surface_','Infiltration Basin',
            'Rain Barrel', 'Time'
        ]
    )
elif args.membership == "exponential":
    physical_interpretation = pd.DataFrame(
        colugenerationsmns = [
            's1', 's2', 's3', 'lambda', 'Runred', 'Cost', 'Pollutant', 
            'Infiltration trench', 'Vegetated_filterstrip', 'Wet_pond', 
            'Bioretention', 'Constructed_wetland', 'Porous_Pavement', 
            'Grassed_swales', 'Sand_filter__surface_','Infiltration Basin',
            'Rain Barrel', 'Time'
        ]
    )
elif args.membership == "nonlinear":
    physical_interpretation = pd.DataFrame(
        columns = [
            'beta1', 'beta2', 'beta3', 'lambda', 'Runred', 'Cost', 'Pollutant', 
            'Infiltration trench', 'Vegetated_filterstrip', 'Wet_pond', 
            'Bioretention', 'Constructed_wetland', 'Porous_Pavement', 
            'Grassed_swales', 'Sand_filter__surface_','Infiltration Basin',
            'Rain Barrel', 'Time'
        ]
    )

alldata = bucketed_data.append(bucketed_rooftop, ignore_index=True)

for i in range(numTrials):
    start_time = time.time()

    if args.membership == "hyperbolic":
        bmp_problem = BMPProblem(membership_function = "hyperbolic")
    elif args.membership == "exponential":
        bmp_problem = BMPProblem(membership_function = "exponential", s = trialParams[i])
    elif args.membership == "nonlinear":
        bmp_problem = BMPProblem(membership_function = "nonlinear", beta = trialParams[i])
    else:
        print("Not supported membership function", args.membership)
        exit(-1)

    if args.model == "GA":
        algorithm = GA(
            pop_size = args.population,
            eliminate_duplicates = True
        )
    else:
        print("Model", args.model, "not found")
        exit(0)

    if args.generations > 0:
        res = minimize(
            bmp_problem,
            algorithm,
            ('n_gen', args.generations),
            seed=args.seed,
            save_history=True,
            verbose=True
        )
    else:
        res = minimize(
            bmp_problem,
            algorithm,
            seed=args.seed,
            save_history=True,
            verbose=True
        )
    end_time = time.time()
    print("Optimization complete for run #", i, "with params =", trialParams[i])
    print("Took %.2f s."%(end_time - start_time))
    print("Results:", res.F, "&", res.X)
    
    x = res.X
    runred = np.sum(alldata["Runredvol"] * x)
    pollutant = np.sum(alldata["Pollutant"] * x)
    cost = np.sum(alldata["Cost"] * x)
    alldata["this_area"] = alldata["Shape_Area"] * x    
    answers = alldata[["BMP", "this_area"]].groupby(["BMP"]).agg("sum")["this_area"].to_dict()

    answers['Runred'] = runred
    answers['Pollutant'] = pollutant
    answers['Cost'] = cost
    answers['Time'] = end_time - start_time
    answers['lambda'] = abs(res.F[0])

    if args.membership == "hyperbolic":
        THIS_RUN_SUFFIX = "default_"
    elif args.membership == "exponential":
        answers['s1'] = trialParams[i][0]
        answers['s2'] = trialParams[i][1]
        answers['s3'] = trialParams[i][2]
        THIS_RUN_SUFFIX = str(answers['s1']) + "_" + str(answers['s2']) + "_" + str(answers['s3']) + "_"
    elif args.membership == "nonlinear":
        answers['beta1'] = trialParams[i][0]
        answers['beta2'] = trialParams[i][1]
        answers['beta3'] = trialParams[i][2]
        THIS_RUN_SUFFIX = str(answers['beta1']) + "_" + str(answers['beta2']) + "_" + str(answers['beta3']) + "_"
    else:
        print("FATAL Error.")
        exit(-1)

    print("Appending:", answers)
    
    physical_interpretation = physical_interpretation.append(answers, ignore_index=True)
    
    if args.membership == "hyperbolic":
        physical_interpretation.to_csv(BASE_DIRECTORY + "hyperbolic.csv")
    elif args.membership == "exponential":
        physical_interpretation.to_csv(BASE_DIRECTORY + "exponential.csv")
    elif args.membership == "nonlinear":
        physical_interpretation.to_csv(BASE_DIRECTORY + "nonlinear.csv")
    
    n_evals = []    # corresponding number of function evaluations
    F = []          # the objective space values in each generation
    cv = []         # constraint violation in each generation

    for algorithm in res.history:
        n_evals.append(algorithm.evaluator.n_eval)
        opt = algorithm.opt
        cv.append(opt.get("CV").min())
        feas = np.where(opt.get("feasible"))[0]
        _F = opt.get("F")[feas]
        F.append(_F)
    print("Processed algorithm results...")
    
    np.savetxt(BASE_DIRECTORY + THIS_RUN_SUFFIX + "X.csv", res.X, delimiter = ",")
    np.savetxt(BASE_DIRECTORY + THIS_RUN_SUFFIX + "F.csv", np.abs(res.F), delimiter = ",")
    with open(BASE_DIRECTORY + THIS_RUN_SUFFIX + "F.pickle", 'wb') as outputfile:
        pickle.dump(F, outputfile, pickle.HIGHEST_PROTOCOL)

    plotF = []
    for f in F:
        plotF.append(abs(np.min(f)))
    
    plt.plot(n_evals, plotF, '-o', markersize=2, linewidth=1)
    plt.title("Optimisation Convergence")
    plt.xlabel("Function Evaluations")
    plt.ylabel("Lambda")
    plt.ticklabel_format(useMathText=True)
    plt.savefig(BASE_DIRECTORY + THIS_RUN_SUFFIX + 'convergence.png', dpi=500)
    plt.close('all')

    with open(BASE_DIRECTORY + THIS_RUN_SUFFIX + "res.pickle", 'wb') as outputfile:
        pickle.dump(res, outputfile, pickle.HIGHEST_PROTOCOL)

## dumping raw data
with open(BASE_DIRECTORY + "info.pickle", 'wb') as outputfile:
    pickle.dump(runtime_information, outputfile, pickle.HIGHEST_PROTOCOL)