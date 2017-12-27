from os import path
import random
import numpy as np
import networkx as nx

from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import MultiGrid
from mesa.time import RandomActivation

from .agent import Human, Wall, FireExit, Furniture, Fire, Door


class FireEvacuation(Model):
    def __init__(self, floor_plan_file, human_count, collaboration_factor, fire_probability, visualise_vision):
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

        # Set up model objects
        self.schedule = RandomActivation(self)
        self.grid = MultiGrid(height, width, torus=False)

        # Used to start a fire at a random furniture location
        self.furniture_list = []

        # Used to easily see if a location is a FireExit or Door, since this needs to be done a lot
        self.fire_exit_list = []
        self.door_list = []

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
                        break

                    self.graph.add_edge(pos, neighbor)

        self.datacollector = DataCollector(
            {"Alive": lambda m: self.count_human_status(m, "alive"),
             "Dead": lambda m: self.count_human_status(m, "dead"),
             "Escaped": lambda m: self.count_human_status(m, "escaped")})

        # Place human agents randomly
        for i in range(0, human_count):
            pos = self.grid.find_empty()

            # Create a random human
            speed = random.randint(1, 2)

            # http://www.who.int/blindness/GLOBALDATAFINALforweb.pdf
            vision_distribution = [0.0058, 0.0365, 0.0424, 0.9153]
            vision = int(np.random.choice(np.arange(1, self.width + 1, (self.width / len(vision_distribution))), p=vision_distribution))

            nervousness = random.randint(1, 10)
            experience = random.randint(1, 10)

            human = Human(pos, speed=speed, vision=vision, collaboration=collaboration_factor, knowledge=0, nervousness=nervousness, role=None, experience=experience, model=self)

            self.grid.place_agent(human, pos)
            self.schedule.add(human)

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
        self.datacollector.collect(self)

        # If there's no fire yet, attempt to start one
        if not self.fire_started:
            self.start_fire()

        # Halt if no more agents alive
        if self.count_human_status(self, "alive") == 0:
            self.running = False

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
