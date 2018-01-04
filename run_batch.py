from mesa.batchrunner import BatchRunner
import argparse
import matplotlib.pyplot as plt
import time
from datetime import timedelta
import sys
import os

from fire_evacuation.model import FireEvacuation
from fire_evacuation.agent import Human

OUTPUT_DIR = "."

MIN_COLLABORATION = 0
MAX_COLLABORATION = 10

GRAPH_DPI = 100
GRAPH_WIDTH = 1920
GRAPH_HEIGHT = 1080

parser = argparse.ArgumentParser()
parser.add_argument("runs", help="Number of repeat runs to do for each parameter setup.", type=int)
parser.add_argument("human_count", help="Number of humans in the simulation.", type=int)


try:
    args = parser.parse_args()
except Exception:
    parser.print_help()
    sys.exit(1)

runs = 1
if args.runs:
    runs = args.runs

human_count = 1
if args.human_count:
    human_count = args.human_count

# Fixed parameters of our batch runs
fixed_params = dict(floor_plan_file="floorplan_2.txt", human_count=human_count, fire_probability=0.8, visualise_vision=False, random_spawn=True, save_plots=True)

# Vary collaboration_factor between MIN and MAX values above
variable_params = dict(collaboration_factor=range(MIN_COLLABORATION, MAX_COLLABORATION + 1))

# At the end of each model run, calculate the percentage of people that escaped
model_reporter = {"PercentageEscaped": lambda m: ((FireEvacuation.count_human_status(m, Human.Status.ESCAPED) / m.human_count) * 100)}

# Create the batch runner
param_run = BatchRunner(FireEvacuation, iterations=runs, variable_parameters=variable_params,
                        fixed_parameters=fixed_params, model_reporters=model_reporter)

print("Running batch test with %i runs for each parameter and %i human agents." % (runs, human_count))
start = time.time()  # Time the batch run

param_run.run_all()  # Run all simulations

end = time.time()
end_timestamp = time.strftime("%Y%m%d-%H%M%S")

elapsed = end - start  # Get the elapsed time in seconds
print("Batch runner finished. Took: %s" % str(timedelta(seconds=elapsed)))

previous_dataframes = [f for f in os.listdir(OUTPUT_DIR) if (os.path.isfile(os.path.join(OUTPUT_DIR, f)) and "dataframe_" in f)]

# Save the dataframe to a file so we have the oppurtunity to concatenate separate dataframes from separate runs
dataframe = param_run.get_model_vars_dataframe()
dataframe.to_pickle(path="dataframe_" + end_timestamp)

# Concatenate any previous dataframes
if previous_dataframes:
    print("Found previous dataframes:", previous_dataframes)

scatter_plot = plt.scatter(dataframe.collaboration_factor, dataframe.PercentageEscaped, figsize=(GRAPH_WIDTH / GRAPH_DPI, GRAPH_HEIGHT / GRAPH_DPI), dpi=GRAPH_DPI)
scatter_plot.set_title("Evacuation Success: " + str(human_count) + " Human Agents")
scatter_plot.set_xlabel("Collaboration Factor")
scatter_plot.set_ylabel("Percentage Escaped")

plt.xlim(MIN_COLLABORATION, MAX_COLLABORATION)
plt.ylim(0, 100)

plt.savefig("batch_run_" + end_timestamp + ".png")
