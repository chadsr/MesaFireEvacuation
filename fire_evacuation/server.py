import os
import numpy as np
from os import listdir, path

from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter

from .model import FireEvacuation
from .agent import FireExit, Wall, Furniture, Fire, Smoke, Human, Sight, Door, DeadHuman


# Creates a visual portrayal of our model in the browser interface
def fire_evacuation_portrayal(agent):
    if agent is None:
        return

    portrayal = {}
    (x, y) = agent.get_position()
    portrayal["x"] = x
    portrayal["y"] = y

    if type(agent) is Human:
        portrayal["scale"] = 1
        portrayal["Layer"] = 5

        if agent.get_mobility() == Human.Mobility.INCAPACITATED:
            # Incapacitated
            portrayal["Shape"] = "fire_evacuation/resources/incapacitated_human.png"
            portrayal["Layer"] = 6
        elif agent.get_mobility() == Human.Mobility.PANIC:
            # Panicked
            portrayal["Shape"] = "fire_evacuation/resources/panicked_human.png"
        elif agent.is_carrying():
            # Carrying someone
            portrayal["Shape"] = "fire_evacuation/resources/carrying_human.png"
        else:
            # Normal
            portrayal["Shape"] = "fire_evacuation/resources/human.png"
    elif type(agent) is Fire:
        portrayal["Shape"] = "fire_evacuation/resources/fire.png"
        portrayal["scale"] = 1
        portrayal["Layer"] = 3
    elif type(agent) is Smoke:
        portrayal["Shape"] = "fire_evacuation/resources/smoke.png"
        portrayal["scale"] = 1
        portrayal["Layer"] = 2
    elif type(agent) is FireExit:
        portrayal["Shape"] = "fire_evacuation/resources/fire_exit.png"
        portrayal["scale"] = 1
        portrayal["Layer"] = 1
    elif type(agent) is Door:
        portrayal["Shape"] = "fire_evacuation/resources/door.png"
        portrayal["scale"] = 1
        portrayal["Layer"] = 1
    elif type(agent) is Wall:
        portrayal["Shape"] = "fire_evacuation/resources/wall.png"
        portrayal["scale"] = 1
        portrayal["Layer"] = 1
    elif type(agent) is Furniture:
        portrayal["Shape"] = "fire_evacuation/resources/furniture.png"
        portrayal["scale"] = 1
        portrayal["Layer"] = 1
    elif type(agent) is DeadHuman:
        portrayal["Shape"] = "fire_evacuation/resources/dead.png"
        portrayal["scale"] = 1
        portrayal["Layer"] = 4
    elif type(agent) is Sight:
        portrayal["Shape"] = "fire_evacuation/resources/eye.png"
        portrayal["scale"] = 0.8
        portrayal["Layer"] = 7

    return portrayal

# Define the charts on our web interface visualisation
status_chart = ChartModule(
    [
        {"Label": "Alive", "Color": "blue"},
        {"Label": "Dead", "Color": "red"},
        {"Label": "Escaped", "Color": "green"},
    ]
)

mobility_chart = ChartModule(
    [
        {"Label": "Normal", "Color": "green"},
        {"Label": "Panic", "Color": "red"},
        {"Label": "Incapacitated", "Color": "blue"},
    ]
)

collaboration_chart = ChartModule(
    [
        {"Label": "Verbal Collaboration", "Color": "orange"},
        {"Label": "Physical Collaboration", "Color": "red"},
        {"Label": "Morale Collaboration", "Color": "pink"},
    ]
)

# Get list of available floorplans
floor_plans = [
    f
    for f in listdir("fire_evacuation/floorplans")
    if path.isfile(path.join("fire_evacuation/floorplans", f))
]

# get the floorplan dimensions to set the grid dimensions
with open(os.path.join("fire_evacuation/floorplans/", floor_plans[0]), "rt") as f:
    floorplan = np.matrix([line.strip().split() for line in f.readlines()])

# Rotate the floorplan so it's interpreted as seen in the text file
floorplan = np.rot90(floorplan, 3)

height, width = np.shape(floorplan)

canvas_element = CanvasGrid(fire_evacuation_portrayal, height, width, 800, 800)
f.close()

# Specify the parameters changeable by the user, in the web interface
model_params = {
    "floor_plan_file": UserSettableParameter(
        "choice", "Floorplan", value=floor_plans[0], choices=floor_plans
    ),
    "human_count": UserSettableParameter("number", "Number Of Human Agents", value=10),
    "collaboration_percentage": UserSettableParameter(
        "slider", "Percentage Collaborating", value=50, min_value=0, max_value=100, step=10
    ),
    "fire_probability": UserSettableParameter(
        "slider", "Probability of Fire", value=0.1, min_value=0, max_value=1, step=0.01
    ),
    "random_spawn": UserSettableParameter(
        "checkbox", "Spawn Agents at Random Locations", value=True
    ),
    "visualise_vision": UserSettableParameter("checkbox", "Show Agent Vision", value=False),
    "save_plots": UserSettableParameter("checkbox", "Save plots to file", value=True),
}

# Start the visual server with the model
server = ModularServer(
    FireEvacuation,
    [canvas_element, status_chart, mobility_chart, collaboration_chart],
    "Fire Evacuation",
    model_params,
)
