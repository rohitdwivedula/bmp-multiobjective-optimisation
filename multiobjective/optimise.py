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

# PyMoo Imports
from pymoo.optimize import minimize
from pymoo.factory import get_sampling
from pymoo.factory import get_termination
from pymoo.factory import get_reference_directions
from pymoo.factory import get_decision_making
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.ctaea import CTAEA
from pymoo.util.ref_dirs.energy import RieszEnergyReferenceDirectionFactory
from pymoo.factory import get_performance_indicator
from pymoo.indicators.hv import Hypervolume
from pymoo.util.running_metric import RunningMetric

parser = argparse.ArgumentParser(description='Run an optimization algorithm on the GHMC BMP data.')

# basic arguments
parser.add_argument('model', help='Name of the model to run it on. Must be one of NSGA2, NSGA3, MOEAD or CTAEA')
parser.add_argument('-d', '--data', type=str, default='historic', help='Run optimisation on historic data or climate change data (future). Must be one of "historic" or "future"')
parser.add_argument('-p', '--population', type=int, default=1000, help='Number of members in the population of the EA')
parser.add_argument('-g', '--generations', type=int, default=10, help='Number of generations to run the EA.')

# reference direction customization
parser.add_argument('-r', '--reference', type=str, default='das-dennis', help='Reference directions for NSGA3 or MOEA/D')
parser.add_argument('-x', '--ref_dirs_count', type=int, default=15, help='Number of directions')

# decomposition customization
parser.add_argument('-t', '--decomposition_type', type=str, default='pbi', help='Decomposition method for MOEA/D')
parser.add_argument('-n', '--neighbours', type=int, default=14, help='Number of neighbours for MOEA/D')
parser.add_argument('-m', '--probability_neighbour', type=float, default=0.7, help='Probability of neihbour mating in MOEA/D')

# deterministic seed
parser.add_argument('-s', '--seed', type=int, default=1, help='Seed for reproducibility')

args = parser.parse_args()

runtime_information = dict()
for arg in vars(args):
    runtime_information[arg] = getattr(args, arg)

print("Begin running with configuration:")
print(runtime_information)

plot_names = dict()
plot_names["NSGA2"] = "NSGA-II"
plot_names["NSGA3"] = "NSGA-III"
plot_names["CTAEA"] = "C-TAEA"

if args.model not in ["NSGA2", "NSGA3", "MOEAD", "CTAEA"]:
    print("Optimization Model", args.model, "is not defined.")
    exit(-1)

if args.data not in ["historic", "future"]:
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



'''
    NADIR POINT

    Nadir point is defined as the least optimal point. In the paper, we used 
    [0, 0, 34972246294.154465] as the Nadir point - corresponding to zero 
    runoff reduction and zero pollutant load reduction, but maximal cost.
'''
NADIR_POINT = [0, 0, np.sum(nonrooftop["Cost"]) + np.sum(rooftop["Cost"])]

rooftop = None
nonrooftop = None
areas = None
bioretention = None
vegetated_strip = None
grassed_swales = None
sand_filter = None
porous_pavement = None
wet_pond = None
constructed_wetland = None

# define BMP problem
class BMPProblem(Problem):
    def __init__(self):
        super().__init__(n_var=bucketed_data.shape[0] + bucketed_rooftop.shape[0],
                         n_obj=3,
                         n_constr=120,
                         xl=lower_limits,
                         xu=upper_limits)
    
    def _evaluate(self, X, out, *args, **kwargs):
        # X is of size (n, bucketed_data.shape[0] + bucketed_rooftop.shape[0]) since we have  
        # that many decision variables
        X_nonroof = X[:,:bucketed_data.shape[0]]
        X_roof = X[:, bucketed_data.shape[0]:]
        
        f1_nonroof = -1 * np.sum(X_nonroof * np.asarray(bucketed_data["Runredvol"]), axis=1)
        f2_nonroof = -1 * np.sum(X_nonroof * np.asarray(bucketed_data["Pollutant"]), axis=1) # multiobjective only
        f3_nonroof = np.sum(X_nonroof * np.asarray(bucketed_data["Cost"]), axis=1)
        
        f1_roof = -1 * np.sum(X_roof * np.asarray(bucketed_rooftop["Runredvol"]), axis=1)
        f2_roof = -1 * np.sum(X_roof * np.asarray(bucketed_rooftop["Pollutant"]), axis=1) # multiobjective only
        f3_roof = np.sum(X_roof * np.asarray(bucketed_rooftop["Cost"]), axis=1)
        
        f1 = f1_nonroof + f1_roof
        f2 = f2_nonroof + f2_roof
        f3 = f3_nonroof + f3_roof
        
        out["F"] = np.column_stack([f1, f2, f3])
        
        constraints = []
        
        for i in range(0, len(all_combinations)):
            c = all_combinations[i]
            upper_limit = all_constraint_values[i]            
            subset = bucketed_data.loc[bucketed_data["BMP"].isin(c)]
            ans = np.sum(np.asarray(subset["Shape_Area"]) * X[:, subset.index], axis = 1) - upper_limit
            constraints.append(ans)
        
        out["G"] = np.column_stack(constraints)


# for saving results
if not os.path.isdir("results/"):
    os.mkdir("results")
previous_runs = list(filter(lambda x: os.path.isdir("results/"+x), os.listdir("results")))
try:
    CURRENT_RUN_NUMBER = max([int(x[3:]) for x in previous_runs]) + 1
except ValueError:
    CURRENT_RUN_NUMBER = 0
BASE_DIRECTORY = f"results/exp{CURRENT_RUN_NUMBER}/"
os.mkdir(BASE_DIRECTORY)

# perform optimization
start_time = time.time()
bmp_problem = BMPProblem()
if args.generations != -1:
	termination = get_termination("n_gen", args.generations)

if args.model == "NSGA2":
    algorithm = NSGA2(
        pop_size=args.population, 
        eliminate_duplicates=True, 
        save_history = True
    )
else:
    if args.reference == "das-dennis":
        ref_dirs = get_reference_directions(args.reference, 3, n_partitions=args.ref_dirs_count)
    elif args.reference == "energy":
        ref_dirs = get_reference_directions(args.reference, 3, args.ref_dirs_count)
    else:
        print("Undefined reference directions:", args.reference)
        exit(-1)
    runtime_information["ref_dirs"] = ref_dirs.shape[0]
    print("Created reference directions... Shape:", ref_dirs.shape)
    if args.model == "NSGA3":
        algorithm = NSGA3(
            pop_size=args.population, 
            ref_dirs=ref_dirs, 
            save_history=True
        )
    elif args.model == "MOEAD":
        algorithm = MOEAD(
            ref_dirs,
            n_neighbors=args.neighbours,
            decomposition=args.decomposition_type,
            prob_neighbor_mating=args.probability_neighbour,
        )
    else:
    	algorithm = CTAEA(
    		ref_dirs
    	)

if args.generations > 0:
	res = minimize(
	    bmp_problem,
	    algorithm,
	    termination,
	    seed=args.seed,
	    sampling=get_sampling("real_random"),
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
print("Optimization complete...")
print("Took %.2f s."%(end_time - start_time))
runtimes["optimization"] = end_time - start_time
runtime_information["time"] = runtimes

# save pareto-front images to file.
x = np.reshape(res.F[:,0:1], newshape=(res.F.shape[0]))
y = np.reshape(res.F[:,1:2], newshape=(res.F.shape[0]))
z = np.reshape(res.F[:,2:3], newshape=(res.F.shape[0]))

def draw_3d_plot(x, y, z, n, normal_points=True, decision_points=False):
    plt.clf()
    fig = plt.figure(figsize = (10,10))

    if decision_points:
        dm = get_decision_making("high-tradeoff")
        I = dm.do(res.F)
        runtime_information["num_high_tradeoff"] = len(I)
    # ax = fig.add_subplot(1, 3, 1, projection='3d')
    # ax.set_xlabel('Runred')
    # ax.set_ylabel('Pollutant')
    # ax.set_zlabel('Cost')
    # if normal_points:
    #     ax.scatter(np.abs(x), np.abs(y), np.abs(z), marker='.')
    # if decision_points:
    #     ax.scatter(np.abs(x)[I], np.abs(y)[I], np.abs(z)[I], marker='o', color='r')

    ax = fig.add_subplot(1, 1, 1, projection='3d')
    if normal_points:
        ax.scatter(np.abs(y), np.abs(z), np.abs(x), marker='.')
    if decision_points:
        ax.scatter(np.abs(y)[I], np.abs(z)[I], np.abs(x)[I], marker='o', color='r')
    
    ax.set_xlabel('Pollutant')
    ax.set_ylabel('Cost')
    ax.set_zlabel('Runred')

    # ax = fig.add_subplot(1, 3, 3, projection='3d')
    # if normal_points:
    #     ax.scatter(np.abs(z), np.abs(x), np.abs(y), marker='.')
    # if decision_points:
    #     ax.scatter(np.abs(z)[I], np.abs(x)[I], np.abs(y)[I], marker='o', color='r')
    # ax.set_xlabel('Cost')
    # ax.set_ylabel('Runred')
    # ax.set_zlabel('Pollutant')
    plt.savefig(BASE_DIRECTORY+'run_{n}_pareto_{a}_{b}.png'.format(
        n=n,
        a=str(int(normal_points)),
        b=str(int(decision_points))
        ) 
    )
    plt.close('all')

draw_3d_plot(x, y, z, CURRENT_RUN_NUMBER, normal_points=True, decision_points=True)
draw_3d_plot(x, y, z, CURRENT_RUN_NUMBER, normal_points=True, decision_points=False)
draw_3d_plot(x, y, z, CURRENT_RUN_NUMBER, normal_points=False, decision_points=True)
plt.close('all')
print("Pareto fronts calculated.")

## Performance Plots - Hypervolume and Running Metric
n_evals = []    # corresponding number of function evaluations
F = []          # the objective space values in each generation
cv = []         # constraint violation in each generation

for algorithm in res.history:
    n_evals.append(algorithm.evaluator.n_eval)
    opt = algorithm.opt
    cv.append(opt.get("CV").min())
    feas = np.where(opt.get("feasible"))[0]
    F.append(opt.get("F")[feas])
print("Processed algorithm results...")

metric = Hypervolume(ref_point=NADIR_POINT, normalize=False)
hv = [metric.do(f) for f in F]
plt.plot(n_evals, hv, '-o', markersize=2, linewidth=1)
plt.title("Convergence")
plt.xlabel("Function Evaluations")
plt.ylabel("Hypervolume")
plt.savefig(BASE_DIRECTORY+'run_{n}_hypervolume.png'.format(n=CURRENT_RUN_NUMBER))
plt.close('all')
print("Hypervolume calculated")
hypervolume = pd.DataFrame({
    "evals": n_evals,
    "hv": hv
})
hypervolume.to_csv(BASE_DIRECTORY+"hv.csv")

## dumping raw data
with open(BASE_DIRECTORY + "info.pickle", 'wb') as outputfile:
    pickle.dump(runtime_information, outputfile, pickle.HIGHEST_PROTOCOL)

with open(BASE_DIRECTORY + "F.pickle", 'wb') as outputfile:
    pickle.dump(F, outputfile, pickle.HIGHEST_PROTOCOL)

# remove unused variables before dumping entire result
print("Delete unused variables")
metric = None
hv = None
running = None
hypervolume = None
runtime_information = None
gc.collect()

np.savetxt(BASE_DIRECTORY + "X.csv", res.X, delimiter = ",")
np.savetxt(BASE_DIRECTORY + "F.csv", np.abs(res.F), delimiter = ",")

alldata = bucketed_data.append(bucketed_rooftop, ignore_index=True)

physical_interpretation = pd.DataFrame(
    columns = ['Runred', 'Cost', 'Pollutant', 'Infiltration trench', 
               'Vegetated_filterstrip', 'Wet_pond', 'Bioretention', 
               'Constructed_wetland', 'Porous_Pavement', 'Grassed_swales', 
               'Sand_filter__surface_', 'Infiltration Basin', 'Rain Barrel']
)

for i in range(0, res.X.shape[0]):
    x = res.X[i]
    runred = np.sum(alldata["Runredvol"] * x)
    pollutant = np.sum(alldata["Pollutant"] * x)
    cost = np.sum(alldata["Cost"] * x)
    try:
        assert (np.round(runred - abs(res.F[i][0])) == 0)
        assert (np.round(pollutant - abs(res.F[i][1])) == 0)
        assert (np.round(cost - abs(res.F[i][2])) == 0)
    except Exception as e:
        print(e)

    alldata["this_area"] = alldata["Shape_Area"] * x
    
    answers = alldata[["BMP", "this_area"]].groupby(["BMP"]).agg("sum")["this_area"].to_dict()
    answers['Runred'] = runred
    answers['Pollutant'] = pollutant
    answers['Cost'] = cost
    physical_interpretation = physical_interpretation.append(answers, ignore_index=True)
    
physical_interpretation.to_csv(BASE_DIRECTORY + "physical_interpretation.csv")

bucketed_data = None
bucketed_rooftop = None

with open(BASE_DIRECTORY + "res.pickle", 'wb') as outputfile:
    pickle.dump(res, outputfile, pickle.HIGHEST_PROTOCOL)

print("Dumped results and runtime information...")
print("All tasks done. Exiting", BASE_DIRECTORY)