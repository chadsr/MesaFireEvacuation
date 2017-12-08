from os import path
import random
import numpy as np
import networkx as nx

from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import Grid
from mesa.time import RandomActivation

from .agent import Human, Wall, FireExit, Furniture


class FireEvacuation(Model):
    def __init__(self, floor_plan_file, human_count, collaboration_factor):
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
        self.fire_started = False  # Turns to true when a fire has started

        # Set up model objects
        self.schedule = RandomActivation(self)
        self.grid = Grid(height, width, torus=False)

        # Load floorplan objects
        for (x, y), value in np.ndenumerate(floorplan):
            value = str(value)
            floor_object = None
            if value is "W":
                floor_object = Wall((x, y), self)
            elif value is "E":
                floor_object = FireExit((x, y), self)
            elif value is "F":
                floor_object = Furniture((x, y), self)

            if floor_object:
                self.grid.place_agent(floor_object, (x, y))
                self.schedule.add(floor_object)

        # Create a graph of traversable routes
        self.graph = nx.Graph()
        for agent, x, y in self.grid.coord_iter():
            pos = (x, y)

            if not agent:
                neighbors = self.grid.get_neighborhood(pos, moore=True, include_center=True, radius=1)
                for neighbor in neighbors:
                    contents = self.grid.get_cell_list_contents(neighbor)
                    if not contents:
                        self.graph.add_edge(pos, neighbor)

        self.datacollector = DataCollector(
            {"Alive": lambda m: self.count_human_status(m, "alive"),
             "Dead": lambda m: self.count_human_status(m, "dead"),
             "Escaped": lambda m: self.count_human_status(m, "escaped")})

        # Place human agents randomly
        for _ in range(0, human_count):
            pos = self.grid.find_empty()

            # Create a random human
            speed = random.randint(1, 2)
            vision = random.randint(1, 30)
            nervousness = random.randint(1, 10)
            experience = random.randint(1, 10)

            human = Human(pos, speed=speed, vision=vision, collaboration=collaboration_factor, knowledge=0, nervousness=nervousness, role=None, experience=experience, model=self)

            self.grid._place_agent(pos, human)
            self.schedule.add(human)

        self.running = True

    def step(self):
        """
        Advance the model by one step.
        """
        self.schedule.step()
        self.datacollector.collect(self)

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
