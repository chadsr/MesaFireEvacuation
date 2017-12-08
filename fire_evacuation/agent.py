from copy import deepcopy

from mesa import Agent, space


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


class FireExit(FloorObject):
    def __init__(self, pos, model):
        self.flammable = False
        self.visibility = 3
        super().__init__(pos, self.flammable, self.visibility, model)


class Wall(FloorObject):
    def __init__(self, pos, model):
        self.flammable = False
        self.visibility = 1
        super().__init__(pos, self.flammable, self.visibility, model)


class Furniture(FloorObject):
    def __init__(self, pos, model):
        self.flammable = True
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

    def step(self):
        pass

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

    def step(self):
        pass

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
        self.pos = pos
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

        # An empty set representing what the agent knows of the floor plan
        self.known_tiles = set()

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
                            visible_neighborhood.add(tile)
                except Exception as e:
                    print(e)

        return visible_neighborhood

        """
        # visualise visible tiles
        for tile in visible_neighborhood:
            if self.model.grid.is_cell_empty(tile):
                sight_object = Sight(tile, self.model)
                self.model.grid.place_agent(sight_object, tile)
                self.model.schedule.add(sight_object)
        """

    def learn_environment(self):
        visible_tiles = self.get_visible_tiles()

        new_tiles = 0
        for tile in visible_tiles:
            if tile not in self.known_tiles:
                self.known_tiles.add(tile)
                new_tiles += 1

        # update the knowledge Attribute accordingly
        total_tiles = self.model.grid.width * self.model.grid.height
        new_knowledge_percentage = new_tiles / total_tiles
        self.knowledge = self.knowledge + new_knowledge_percentage
        print("Current knowledge:", self.knowledge)

    def step(self):
        self.learn_environment()

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
