import os
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time

from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import Coordinate, MultiGrid
from mesa.time import RandomActivation

from .agent import Human, Wall, FireExit, Furniture, Fire, Door


class FireEvacuation(Model):
    MIN_HEALTH = 0.75
    MAX_HEALTH = 1

    MIN_SPEED = 1
    MAX_SPEED = 2

    MIN_NERVOUSNESS = 1
    MAX_NERVOUSNESS = 10

    MIN_EXPERIENCE = 1
    MAX_EXPERIENCE = 10

    MIN_VISION = 1
    # MAX_VISION is simply the size of the grid

    def __init__(
        self,
        floor_plan_file: str,
        human_count: int,
        collaboration_percentage: float,
        fire_probability: float,
        visualise_vision: bool,
        random_spawn: bool,
        save_plots: bool,
    ):
        # Load floorplan
        # floorplan = np.genfromtxt(path.join("fire_evacuation/floorplans/", floor_plan_file))
        with open(os.path.join("fire_evacuation/floorplans/", floor_plan_file), "rt") as f:
            floorplan = np.matrix([line.strip().split() for line in f.readlines()])

        # Rotate the floorplan so it's interpreted as seen in the text file
        floorplan = np.rot90(floorplan, 3)

        # Check what dimension our floorplan is
        width, height = np.shape(floorplan)

        # Init params
        self.width = width
        self.height = height
        self.human_count = human_count
        self.collaboration_percentage = collaboration_percentage
        self.visualise_vision = visualise_vision
        self.fire_probability = fire_probability
        self.fire_started = False  # Turns to true when a fire has started
        self.save_plots = save_plots

        # Set up model objects
        self.schedule = RandomActivation(self)

        self.grid = MultiGrid(height, width, torus=False)

        # Used to start a fire at a random furniture location
        self.furniture: dict[Coordinate, Furniture] = {}

        # Used to easily see if a location is a FireExit or Door, since this needs to be done a lot
        self.fire_exits: dict[Coordinate, FireExit] = {}
        self.doors: dict[Coordinate, Door] = {}

        # If random spawn is false, spawn_pos_list will contain the list of possible spawn points according to the floorplan
        self.random_spawn = random_spawn
        self.spawn_pos_list: list[Coordinate] = []

        # Load floorplan objects
        for (x, y), value in np.ndenumerate(floorplan):
            pos: Coordinate = (x, y)

            value = str(value)
            floor_object = None
            if value == "W":
                floor_object = Wall(pos, self)
            elif value == "E":
                floor_object = FireExit(pos, self)
                self.fire_exits[pos] = floor_object
                # Add fire exits to doors as well, since, well, they are
                self.doors[pos] = floor_object
            elif value == "F":
                floor_object = Furniture(pos, self)
                self.furniture[pos] = floor_object
            elif value == "D":
                floor_object = Door(pos, self)
                self.doors[pos] = floor_object
            elif value == "S":
                self.spawn_pos_list.append(pos)

            if floor_object:
                self.grid.place_agent(floor_object, pos)
                self.schedule.add(floor_object)

        # Create a graph of traversable routes, used by agents for pathing
        self.graph = nx.Graph()
        for agents, x, y in self.grid.coord_iter():
            pos = (x, y)

            # If the location is empty, or a door
            if not agents or any(isinstance(agent, Door) for agent in agents):
                neighbors = self.grid.get_neighborhood(
                    pos, moore=True, include_center=True, radius=1
                )

                for neighbor in neighbors:
                    # If there is contents at this location and they are not Doors or FireExits, skip them
                    if not self.grid.is_cell_empty(neighbor) and neighbor not in self.doors.keys():
                        continue

                    self.graph.add_edge(pos, neighbor)

        # Collects statistics from our model run
        self.datacollector = DataCollector(
            {
                "Alive": lambda m: self.count_human_status(m, Human.Status.ALIVE),
                "Dead": lambda m: self.count_human_status(m, Human.Status.DEAD),
                "Escaped": lambda m: self.count_human_status(m, Human.Status.ESCAPED),
                "Incapacitated": lambda m: self.count_human_mobility(
                    m, Human.Mobility.INCAPACITATED
                ),
                "Normal": lambda m: self.count_human_mobility(m, Human.Mobility.NORMAL),
                "Panic": lambda m: self.count_human_mobility(m, Human.Mobility.PANIC),
                "Verbal Collaboration": lambda m: self.count_human_collaboration(
                    m, Human.Action.VERBAL_SUPPORT
                ),
                "Physical Collaboration": lambda m: self.count_human_collaboration(
                    m, Human.Action.PHYSICAL_SUPPORT
                ),
                "Morale Collaboration": lambda m: self.count_human_collaboration(
                    m, Human.Action.MORALE_SUPPORT
                ),
            }
        )

        # Calculate how many agents will be collaborators
        number_collaborators = int(round(self.human_count * (self.collaboration_percentage / 100)))

        # Start placing human agents
        for i in range(0, self.human_count):
            if self.random_spawn:  # Place human agents randomly
                pos = self.grid.find_empty()
            else:  # Place human agents at specified spawn locations
                pos = random.choice(self.spawn_pos_list)

            if pos:
                # Create a random human
                health = random.randint(self.MIN_HEALTH * 100, self.MAX_HEALTH * 100) / 100
                speed = random.randint(self.MIN_SPEED, self.MAX_SPEED)

                if number_collaborators > 0:
                    collaborates = True
                    number_collaborators -= 1
                else:
                    collaborates = False

                # Vision statistics obtained from http://www.who.int/blindness/GLOBALDATAFINALforweb.pdf
                vision_distribution = [0.0058, 0.0365, 0.0424, 0.9153]
                vision = int(
                    np.random.choice(
                        np.arange(
                            self.MIN_VISION,
                            self.width + 1,
                            (self.width / len(vision_distribution)),
                        ),
                        p=vision_distribution,
                    )
                )

                nervousness_distribution = [
                    0.025,
                    0.025,
                    0.1,
                    0.1,
                    0.1,
                    0.3,
                    0.2,
                    0.1,
                    0.025,
                    0.025,
                ]  # Distribution with slight higher weighting for above median nerovusness
                nervousness = int(
                    np.random.choice(
                        range(self.MIN_NERVOUSNESS, self.MAX_NERVOUSNESS + 1),
                        p=nervousness_distribution,
                    )
                )  # Random choice starting at 1 and up to and including 10

                experience = random.randint(self.MIN_EXPERIENCE, self.MAX_EXPERIENCE)

                belief_distribution = [0.9, 0.1]  # [Believes, Doesn't Believe]
                believes_alarm = np.random.choice([True, False], p=belief_distribution)

                human = Human(
                    pos,
                    health=health,
                    speed=speed,
                    vision=vision,
                    collaborates=collaborates,
                    nervousness=nervousness,
                    experience=experience,
                    believes_alarm=believes_alarm,
                    model=self,
                )

                self.grid.place_agent(human, pos)
                self.schedule.add(human)
            else:
                print("No tile empty for human placement!")

        self.running = True

    # Plots line charts of various statistics from a run
    def save_figures(self):
        DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        OUTPUT_DIR = DIR + "/output"

        results = self.datacollector.get_model_vars_dataframe()

        dpi = 100
        fig, axes = plt.subplots(figsize=(1920 / dpi, 1080 / dpi), dpi=dpi, nrows=1, ncols=3)

        status_results = results.loc[:, ["Alive", "Dead", "Escaped"]]
        status_plot = status_results.plot(ax=axes[0])
        status_plot.set_title("Human Status")
        status_plot.set_xlabel("Simulation Step")
        status_plot.set_ylabel("Count")

        mobility_results = results.loc[:, ["Incapacitated", "Normal", "Panic"]]
        mobility_plot = mobility_results.plot(ax=axes[1])
        mobility_plot.set_title("Human Mobility")
        mobility_plot.set_xlabel("Simulation Step")
        mobility_plot.set_ylabel("Count")

        collaboration_results = results.loc[
            :, ["Verbal Collaboration", "Physical Collaboration", "Morale Collaboration"]
        ]
        collaboration_plot = collaboration_results.plot(ax=axes[2])
        collaboration_plot.set_title("Human Collaboration")
        collaboration_plot.set_xlabel("Simulation Step")
        collaboration_plot.set_ylabel("Successful Attempts")
        collaboration_plot.set_ylim(ymin=0)

        timestr = time.strftime("%Y%m%d-%H%M%S")
        plt.suptitle(
            "Percentage Collaborating: "
            + str(self.collaboration_percentage)
            + "%, Number of Human Agents: "
            + str(self.human_count),
            fontsize=16,
        )
        plt.savefig(OUTPUT_DIR + "/model_graphs/" + timestr + ".png")
        plt.close(fig)

    # Starts a fire at a random piece of furniture with file_probability chance
    def start_fire(self):
        rand = random.random()
        if rand < self.fire_probability:
            fire_furniture: Furniture = random.choice(list(self.furniture.values()))
            pos = fire_furniture.pos

            fire = Fire(pos, self)
            self.grid.place_agent(fire, pos)
            self.schedule.add(fire)

            self.fire_started = True
            print(f"Fire started at position {pos}")

    def step(self):
        """
        Advance the model by one step.
        """

        self.schedule.step()

        # If there's no fire yet, attempt to start one
        if not self.fire_started:
            self.start_fire()

        self.datacollector.collect(self)

        # If no more agents are alive, stop the model and collect the results
        if self.count_human_status(self, Human.Status.ALIVE) == 0:
            self.running = False

            if self.save_plots:
                self.save_figures()

    @staticmethod
    def count_human_collaboration(model, collaboration_type):
        """
        Helper method to count the number of collaborations performed by Human agents in the model
        """

        count = 0
        for agent in model.schedule.agents:
            if isinstance(agent, Human):
                if collaboration_type == Human.Action.VERBAL_SUPPORT:
                    count += agent.get_verbal_collaboration_count()
                elif collaboration_type == Human.Action.MORALE_SUPPORT:
                    count += agent.get_morale_collaboration_count()
                elif collaboration_type == Human.Action.PHYSICAL_SUPPORT:
                    count += agent.get_physical_collaboration_count()

        return count

    @staticmethod
    def count_human_status(model, status):
        """
        Helper method to count the status of Human agents in the model
        """
        count = 0
        for agent in model.schedule.agents:
            if isinstance(agent, Human) and agent.get_status() == status:
                count += 1

        return count

    @staticmethod
    def count_human_mobility(model, mobility):
        """
        Helper method to count the mobility of Human agents in the model
        """
        count = 0
        for agent in model.schedule.agents:
            if isinstance(agent, Human) and agent.get_mobility() == mobility:
                count += 1

        return count
