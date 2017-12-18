import networkx as nx
import numpy as np
import sys
import random
from copy import deepcopy

from mesa import Agent


# Credits to http://www.roguebasin.com/index.php?title=Bresenham%27s_Line_Algorithm
# Can be re-written eventually
def get_line(start, end):
    """Bresenham's Line Algorithm
    Produces a list of tuples from start and end

    >>> points1 = get_line((0, 0), (3, 4))
    >>> points2 = get_line((3, 4), (0, 0))
    >>> assert(set(points1) == set(points2))
    >>> print points1
    [(0, 0), (1, 1), (1, 2), (2, 3), (3, 4)]
    >>> print points2
    [(3, 4), (2, 3), (1, 2), (1, 1), (0, 0)]
    """
    # Setup initial conditions
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1

    # Determine how steep the line is
    is_steep = abs(dy) > abs(dx)

    # Rotate line
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    # Swap start and end points if necessary and store swap state
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True

    # Recalculate differentials
    dx = x2 - x1
    dy = y2 - y1

    # Calculate error
    error = int(dx / 2.0)
    ystep = 1 if y1 < y2 else -1

    # Iterate over bounding box generating points between start and end
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = (y, x) if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += ystep
            error += dx

    # Reverse the list if the coordinates were swapped
    if swapped:
        points.reverse()
    return points


class Sight(Agent):
    def __init__(self, pos, model):
        super().__init__(pos, model)
        self.pos = pos

    def get_position(self):
        return self.pos


"""
FLOOR STUFF
"""


class FloorObject(Agent):
    def __init__(self, pos, flammable, visibility, model):
        super().__init__(pos, model)
        self.pos = pos

    def get_position(self):
        return self.pos


class Door(FloorObject):
    def __init__(self, pos, model):
        self.flammable = False
        self.spreads_smoke = True
        self.visibility = 1
        super().__init__(pos, self.flammable, self.visibility, model)


class FireExit(FloorObject):
    def __init__(self, pos, model):
        self.flammable = False
        self.spreads_smoke = False
        self.visibility = 3
        super().__init__(pos, self.flammable, self.visibility, model)


class Wall(FloorObject):
    def __init__(self, pos, model):
        self.flammable = False
        self.spreads_smoke = False
        self.visibility = 1
        super().__init__(pos, self.flammable, self.visibility, model)


class Furniture(FloorObject):
    def __init__(self, pos, model):
        self.flammable = True
        self.spreads_smoke = True
        self.visibility = 1
        super().__init__(pos, self.flammable, self.visibility, model)


"""
FIRE STUFF
"""


class Fire(Agent):
    """
    A fire agent

    Attributes:
        ...
    """

    def __init__(self, pos, model):
        super().__init__(pos, model)
        self.pos = pos
        self.smoke_radius = 1
        self.flammable = False
        self.spreads_smoke = True

    def step(self):
        neighborhood = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False, radius=self.smoke_radius)

        for neighbor in neighborhood:
            place_smoke = True
            place_fire = True
            contents = self.model.grid.get_cell_list_contents(neighbor)

            if contents:
                for agent in contents:
                    print("Agent:", agent, agent.spreads_smoke, agent.flammable)
                    if not agent.flammable:
                        place_fire = False
                        break
                    if not agent.spreads_smoke:
                        place_smoke = False
                        break
            else:
                place_fire = False

            print(neighbor, "Fire:", place_fire, "Smoke:", place_smoke)

            if place_fire:
                print("Place Fire")
                fire = Fire(neighbor, self.model)
                self.model.grid.place_agent(fire, neighbor)
                self.model.schedule.add(fire)
            if place_smoke:
                print("Place Smoke")
                smoke = Smoke(neighbor, self.model)
                self.model.grid.place_agent(smoke, neighbor)
                self.model.schedule.add(smoke)

    def get_position(self):
        return self.pos


class Smoke(Agent):
    """
    A smoke agent

    Attributes:
        ...
    """

    def __init__(self, pos, model):
        super().__init__(pos, model)
        self.pos = pos
        self.smoke_radius = 1
        self.flammable = False
        self.spreads_smoke = False

    def step(self):
        smoke_neighborhood = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False, radius=self.smoke_radius)
        for neighbor in smoke_neighborhood:
            place_smoke = True
            contents = self.model.grid.get_cell_list_contents(neighbor)
            for agent in contents:
                if not agent.spreads_smoke:
                    place_smoke = False

            if place_smoke:
                smoke = Smoke(neighbor, self.model)
                self.model.grid.place_agent(smoke, neighbor)
                self.model.schedule.add(smoke)

    def get_position(self):
        return self.pos


class DeadHuman(Agent):
    def __init__(self, pos, model):
        super().__init__(pos, model)
        self.pos = pos
        self.flammable = False
        self.spreads_smoke = True

    def get_position(self):
        return self.pos


class Human(Agent):
    """
    A human agent, which will attempt to escape from the grid.

    Attributes:
        ID: Unique identifier of the Agent
        Position (x,y): Position of the agent on the Grid
        Health: Health of the agent (between 0 and 1)
        ...
    """

    def __init__(self, pos, speed, vision, collaboration, knowledge, nervousness, role, experience, model):
        super().__init__(pos, model)

        self.flammable = True
        self.spreads_smoke = True

        self.pos = pos
        self.previous_pos = None
        self.health = 1.0
        self.mobility = 1
        self.speed = speed
        self.vision = vision
        self.collaboration = collaboration
        self.knowledge = knowledge
        self.nervousness = nervousness
        self.role = role
        self.experience = experience
        self.escaped = False
        self.planned_target = None  # The location (agent, (x, y)) the agent is planning to move to

        # An empty set representing what the agent knows of the floor plan
        self.known_tiles = set()

        # A set representing where the agent has between
        self.visited_tiles = {self.pos}

    # A strange implementation of ray-casting, using Bresenham's Line Algorithm
    def get_visible_tiles(self):
        neighborhood = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=True, radius=self.vision)
        visible_neighborhood = set()

        for pos in neighborhood:
            wall = False
            if not self.model.grid.out_of_bounds(pos):
                try:
                    path = get_line(self.pos, pos)
                    for tile in path:
                        contents = self.model.grid.get_cell_list_contents(tile)
                        for obj in contents:
                            if isinstance(obj, Wall):  # We hit a wall, reject rest of path and move to next
                                wall = True
                                break
                        if wall:
                            break
                        else:
                            visible_neighborhood.add((tuple(contents), tile))
                except Exception as e:
                    print(e)

        if self.model.visualise_vision:
            for _, tile in visible_neighborhood:
                if self.model.grid.is_cell_empty(tile):
                    sight_object = Sight(tile, self.model)
                    self.model.grid.place_agent(sight_object, tile)

        return visible_neighborhood

    def get_random_target(self, allow_visited=True):
        graph_nodes = self.model.graph.nodes()

        tiles = self.known_tiles

        # If we are excluding visited tiles, remove the visited_tiles set from the available tiles
        if not allow_visited:
            tiles -= self.visited_tiles

        while not self.planned_target:
            target_contents, target_pos = random.choice(list(tiles))
            if target_pos in graph_nodes and target_pos != self.pos:
                self.planned_target = target_contents, target_pos

    def attempt_exit_plan(self, visible_tiles):
        self.planned_target = None
        fire_exits = set()

        for contents, pos in self.known_tiles:
            for agent in contents:
                if isinstance(agent, FireExit):
                    fire_exits.add((agent, pos))

        if fire_exits:
            if len(fire_exits) > 1:  # If there is more than one exit known
                best_distance = None
                for exit, exit_pos in fire_exits:
                    length = len(get_line(self.pos, exit_pos))  # Let's use Bresenham's to find the 'closest' exit
                    if not best_distance or length < best_distance:
                        best_distance = length
                        self.planned_target = exit, exit_pos

            else:
                self.planned_target = fire_exits.pop()

            print("Agent found a fire escape!", self.planned_target)
        else:  # If there's a fire and no fire-escape in sight, try to head for an unvisited door, if no door in sight, move randomly (for now)
            for contents, pos in visible_tiles:
                for agent in contents:
                    if isinstance(agent, Door) and pos not in self.visited_tiles:
                        # print("FOUND NEW DOOR IN SIGHT INSTEAD")
                        self.planned_target = agent, pos
                        break
                    elif self.planned_target:
                        break

            # Still didn't find a planned_target, so get a random unvisited target
            if not self.planned_target:
                self.get_random_target(allow_visited=False)

    def get_panic_score(self):
        health_component = (1 / np.exp(self.health * 4))
        experience_component = (1 / np.exp(self.experience / 2))
        nervousness_component = (self.nervousness / 10)
        panic_score = health_component * experience_component * nervousness_component

        return panic_score

    def health_mobility_rules(self):
        moore_neighborhood = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=True, radius=1)
        contents = self.model.grid.get_cell_list_contents(moore_neighborhood)
        for agent in contents:
            if isinstance(agent, Fire):
                self.health -= 0.2
                self.speed -= 2
                print("Agent got burnt!")
            elif isinstance(agent, Smoke):
                self.health -= 0.1
                self.speed -= 1
                print("Agent got hurt by smoke!")

        # Prevent health and speed from going below 0
        if self.health < 0:
            self.health = 0
        if self.speed < 0:
            self.speed = 0

        if self.health == 0:
            print(self.pos)
            dead_self = DeadHuman(self.pos, self.model)
            self.model.grid.place_agent(dead_self, self.pos)
            self.model.grid.remove_agent(self)
        elif self.speed == 0:
            self.mobility = 0

    def panic_rules(self):
        panic_score = self.get_panic_score()
        # print("Panic score:", panic_score)

        if panic_score > 0.75:
            print("Agent is panicking!")
            self.mobility = 2

    def learn_environment(self):
        visible_tiles = self.get_visible_tiles()

        new_tiles = 0
        for agent, pos in visible_tiles:
            if (agent, pos) not in self.known_tiles:
                self.known_tiles.add((agent, pos))
                new_tiles += 1

        # update the knowledge Attribute accordingly
        total_tiles = self.model.grid.width * self.model.grid.height
        new_knowledge_percentage = new_tiles / total_tiles
        self.knowledge = self.knowledge + new_knowledge_percentage
        # print("Current knowledge:", self.knowledge)

        return visible_tiles

    def check_for_collaboration(self, visible_tiles):
        for agent, location in visible_tiles:
            if isinstance(agent, Human):
                # Physical/Morale collaboration
                pass
            elif isinstance(agent, FireExit):
                # Verbal collaboration
                pass

    def get_next_location(self, path):
        try:
            length = len(path)
            if length <= self.speed:
                return path[length - 1]
            else:
                return path[self.speed]
        except Exception as e:
            print("Failed to get next location:", e, "\nPath:", path, length, "Speed:", self.speed)
            sys.exit(1)

    def tile_available(self, tile):
        pass

    def get_path(self, graph, visible_tiles, target):
        # If the target location is visible, do a shortest path. else roughly wander in the right direction
        planned_contents, planned_pos = self.planned_target

        try:
            if planned_pos in visible_tiles:
                # print("PLANNED POS IS VISIBLE")
                return nx.shortest_path(graph, self.pos, planned_pos)
            else:
                # print("PLANNED POS IS NOT VISIBLE")
                # TODO: Replace with something more humanly (less efficient)
                return nx.shortest_path(graph, self.pos, planned_pos)
        except nx.exception.NodeNotFound as e:
            graph_nodes = graph.nodes()

            if planned_pos not in graph_nodes:
                print("Target node not found!", planned_pos, planned_contents)
                return None
            elif self.pos not in graph_nodes:
                print("Current position not found!", self.pos)
                return None
            else:
                print(e)
                sys.exit(1)
        except nx.exception.NetworkXNoPath as e:
            print("No path between nodes!", self.pos, planned_pos, planned_contents)
            return None

    def move_toward_target(self, visible_tiles):
        next_location = None
        graph = self.model.graph

        while self.planned_target and not next_location:
            path = self.get_path(graph, visible_tiles, self.planned_target)

            if path:
                next_location = self.get_next_location(path)

                if self.model.grid.is_cell_empty(next_location) or next_location in self.model.door_list:
                    self.previous_pos = self.pos
                    self.model.grid.move_agent(self, next_location)
                    self.visited_tiles.add(next_location)
                else:
                    print("Cell not empty!\nLocation:", self.pos, "\nNext Location:", next_location, "\nTarget:", self.planned_target, "\nPath:", path, "\n")
                    graph = deepcopy(graph)  # Make a deepcopy so we can edit it without affecting the original graph
                    graph.remove_node(next_location)  # Remove the next location from the temporary graph so we can try pathing again without it
                    print("Removed:", next_location)
                    # Next location is blocked, so find another route to target
                    next_location = None

                _, planned_pos = self.planned_target
                if self.pos == planned_pos:
                    # The human reached their target!
                    self.planned_target = None
            else:
                print("Target dropped")
                self.planned_target = None

    def step(self):
        if not self.escaped and self.pos:
            self.health_mobility_rules()

            if self.health > 0:
                self.panic_rules()
                visible_tiles = self.learn_environment()

                # If a fire has started, attempt to plan an exit location
                if self.model.fire_started:
                    self.attempt_exit_plan(visible_tiles)

                # Check if anything in vision can be collaborated with
                self.check_for_collaboration(visible_tiles)

                if not self.planned_target:
                    self.get_random_target()

                if self.mobility == 1:
                    self.move_toward_target(visible_tiles)
                elif self.mobility == 2:  # Panic movement
                    pass

                # Agent reached a fire escape, proceed to exit
                if self.pos in self.model.fire_exit_list:
                    self.escaped = True
                    self.model.grid.remove_agent(self)

    def get_status(self):
        if self.health > 0 and not self.escaped:
            return "alive"
        elif self.health <= 0 and not self.escaped:
            return "dead"
        elif self.escaped:
            return "escaped"

        return None

    def get_speed(self):
        return self.speed

    def get_mobility(self):
        return self.mobility

    def get_health(self):
        return self.health

    def get_position(self):
        return self.pos
