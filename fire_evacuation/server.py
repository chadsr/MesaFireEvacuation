from os import listdir, path

from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter

from .model import FireEvacuation
from .agent import FireExit, Wall, Furniture, Fire, Smoke, Human, Sight, Door, DeadHuman


def fire_evacuation_portrayal(agent):
    if agent is None:
        return

    portrayal = {}
    (x, y) = agent.get_position()
    portrayal["x"] = x
    portrayal["y"] = y

    if type(agent) is Human:
        portrayal["scale"] = 1
        portrayal["Layer"] = 3

        if agent.get_mobility() == 0:  # Incapacitated
            portrayal["Shape"] = "fire_evacuation/resources/incapacitated_human.png"
        elif agent.get_mobility() == 2:  # Panicked
            portrayal["Shape"] = "fire_evacuation/resources/panicked_human.png"
        else:  # Normal
            portrayal["Shape"] = "fire_evacuation/resources/human.png"
    elif type(agent) is Fire:
        portrayal["Shape"] = "fire_evacuation/resources/fire.png"
        portrayal["scale"] = 1
        portrayal["Layer"] = 2
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
        portrayal["Layer"] = 3
    elif type(agent) is Sight:
        portrayal["Shape"] = "fire_evacuation/resources/eye.png"
        portrayal["scale"] = 0.8
        portrayal["Layer"] = 0

    return portrayal


# Was hoping floorplan could dictate the size of the grid, but seems the grid needs to be specified first :/
canvas_element = CanvasGrid(fire_evacuation_portrayal, 50, 50, 800, 800)

status_chart = ChartModule([{"Label": "Alive", "Color": "blue"},
                            {"Label": "Dead", "Color": "red"},
                            {"Label": "Escaped", "Color": "green"}])

mobility_chart = ChartModule([{"Label": "Normal", "Color": "green"},
                              {"Label": "Panic", "Color": "red"},
                              {"Label": "Incapacitated", "Color": "blue"}])

# Get list of available floorplans
floor_plans = [f for f in listdir("fire_evacuation/floorplans") if path.isfile(path.join("fire_evacuation/floorplans", f))]

model_params = {
    "floor_plan_file": UserSettableParameter("choice", "Floorplan", value=floor_plans[0], choices=floor_plans),
    "human_count": UserSettableParameter("number", "Number Of Human Agents", value=10),
    "collaboration_factor": UserSettableParameter("slider", "Collaboration Factor", value=10, min_value=0, max_value=10, step=1),
    "fire_probability": UserSettableParameter("slider", "Probability of Fire", value=0.1, min_value=0, max_value=1, step=0.01),
    "random_spawn": UserSettableParameter('checkbox', 'Spawn Agents at Random Locations', value=True),
    "multithreaded": UserSettableParameter('checkbox', 'Use Multithreading', value=False),
    "visualise_vision": UserSettableParameter('checkbox', 'Show Agent Vision', value=False)
}
server = ModularServer(FireEvacuation, [canvas_element, status_chart, mobility_chart], "Fire Evacuation",
                       model_params)
