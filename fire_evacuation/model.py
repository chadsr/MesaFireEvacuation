from os import path
import random
import numpy as np
import networkx as nx

from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import MultiGrid
from mesa.time import RandomActivation

from .scheduler import MultithreadedRandomActivation
from .agent import Human, Wall, FireExit, Furniture, Fire, Door

MIN_HEALTH = 75
MAX_HEALTH = 100

MIN_SPEED = 1
MAX_SPEED = 2

MIN_NERVOUSNESS = 1
MAX_NERVOUSNESS = 10

MIN_EXPERIENCE = 1
MAX_EXPERIENCE = 10

MIN_VISION = 1
# MAX_VISION is simply the size of the grid


class FireEvacuation(Model):
    def __init__(self, floor_plan_file, human_count, collaboration_factor, fire_probability, visualise_vision, random_spawn, multithreaded):
        # Load floorplan
        # floorplan = np.genfromtxt(path.join("fire_evacuation/floorplans/", floor_plan_file))
        with open(path.join("fire_evacuation/floorplans/", floor_plan_file), "rt") as f:
            floorplan = np.matrix([line.strip().split() for line in f.readlines()])

        # Rotate the floorplan so it's interpreted as seen in the text file
        floorplan = np.rot90(floorplan, 3)

        # Check what dimension our floorplan is
        width, height = np.shape(floorplan)

        # Init params
        self.width = width
        self.height = height
        self.human_count = human_count
        self.collaboration_factor = collaboration_factor
        self.visualise_vision = visualise_vision
        self.fire_probability = fire_probability
        self.fire_started = False  # Turns to true when a fire has started
        self.finished = False

        # Set up model objects
        if multithreaded:
            self.schedule = MultithreadedRandomActivation(self)
        else:
            self.schedule = RandomActivation(self)

        self.grid = MultiGrid(height, width, torus=False)

        # Used to start a fire at a random furniture location
        self.furniture_list = []

        # Used to easily see if a location is a FireExit or Door, since this needs to be done a lot
        self.fire_exit_list = []
        self.door_list = []
        self.random_spawn = random_spawn
        self.spawn_list = []

        # Load floorplan objects
        for (x, y), value in np.ndenumerate(floorplan):
            value = str(value)
            floor_object = None
            if value is "W":
                floor_object = Wall((x, y), self)
            elif value is "E":
                floor_object = FireExit((x, y), self)
                self.fire_exit_list.append((x, y))
                self.door_list.append((x, y))  # Add fire exits to doors as well, since, well, they are
            elif value is "F":
                floor_object = Furniture((x, y), self)
                self.furniture_list.append((x, y))
            elif value is "D":
                floor_object = Door((x, y), self)
                self.door_list.append((x, y))
            elif value is "S":
                self.spawn_list.append((x, y))

            if floor_object:
                self.grid.place_agent(floor_object, (x, y))
                self.schedule.add(floor_object)

        # Create a graph of traversable routes
        self.graph = nx.Graph()
        for agents, x, y in self.grid.coord_iter():
            pos = (x, y)

            # If the location is empty, or a door
            if not agents or any(isinstance(agent, Door) for agent in agents):
                neighbors = self.grid.get_neighborhood(pos, moore=True, include_center=True, radius=1)

                for neighbor in neighbors:
                    # If there is contents at this location and they are not Doors or FireExits, skip them
                    if not self.grid.is_cell_empty(neighbor) and neighbor not in self.door_list:
                        continue

                    self.graph.add_edge(pos, neighbor)

        self.datacollector = DataCollector(
            {
                "Alive": lambda m: self.count_human_status(m, "alive"),
                "Dead": lambda m: self.count_human_status(m, "dead"),
                "Escaped": lambda m: self.count_human_status(m, "escaped"),
                "Incapacitated": lambda m: self.count_human_mobility(m, 0),
                "Normal": lambda m: self.count_human_mobility(m, 1),
                "Panic": lambda m: self.count_human_mobility(m, 2)
            }
        )

        for i in range(0, human_count):
            if self.random_spawn:  # Place human agents randomly
                pos = self.grid.find_empty()
            else:  # Place human agents at specified spawn locations
                pos = random.choice(self.spawn_list)

            if pos:
                # Create a random human
                health = random.randint(MIN_HEALTH, MAX_HEALTH) / 100
                speed = random.randint(MIN_SPEED, MAX_SPEED)

                # http://www.who.int/blindness/GLOBALDATAFINALforweb.pdf
                vision_distribution = [0.0058, 0.0365, 0.0424, 0.9153]
                vision = int(np.random.choice(np.arange(MIN_VISION, self.width + 1, (self.width / len(vision_distribution))), p=vision_distribution))

                nervousness_distribution = [0.025, 0.025, 0.1, 0.1, 0.1, 0.3, 0.2, 0.1, 0.025, 0.025]  # Distribution with slight higher weighting for above median nerovusness
                nervousness = int(np.random.choice(range(MIN_NERVOUSNESS, MAX_NERVOUSNESS + 1), p=nervousness_distribution))  # Random choice starting at 1 and up to and including 10
                experience = random.randint(MIN_EXPERIENCE, MAX_EXPERIENCE)

                belief_distribution = [0.9, 0.1]  # [Believes, Doesn't Believe]
                believes_alarm = np.random.choice([True, False], p=belief_distribution)

                human = Human(pos, health=health, speed=speed, vision=vision, collaboration=collaboration_factor, knowledge=0, nervousness=nervousness, role=None, experience=experience, believes_alarm=believes_alarm, model=self)

                self.grid.place_agent(human, pos)
                self.schedule.add(human)
            else:
                print("No tile empty for human placement!")

        self.running = True

    def start_fire(self):
        rand = random.random()
        if rand < self.fire_probability:
            fire_furniture = random.choice(self.furniture_list)
            fire = Fire(fire_furniture, self)
            self.grid.place_agent(fire, fire_furniture)
            self.schedule.add(fire)
            self.fire_started = True
            print("Fire started at:", fire_furniture)

    def step(self):
        """
        Advance the model by one step.
        """

        self.schedule.step()

        # If there's no fire yet, attempt to start one
        if not self.fire_started:
            self.start_fire()

        self.datacollector.collect(self)

        if self.finished:
            self.running = False

        # If no more agents are alive, stop the model after the next step
        if self.count_human_status(self, "alive") == 0:
            self.finished = True

    @staticmethod
    def count_human_status(model, status):
        """
        Helper method to count the status of Human agents in the model
        """
        count = 0
        for agent in model.schedule.agents:
            if isinstance(agent, Human):
                if agent.get_status() == status:
                    count += 1
        return count

    @staticmethod
    def count_human_mobility(model, mobility):
        """
        Helper method to count the status of Human agents in the model
        """
        count = 0
        for agent in model.schedule.agents:
            if isinstance(agent, Human):
                if agent.get_mobility() == mobility:
                    count += 1
        return count
