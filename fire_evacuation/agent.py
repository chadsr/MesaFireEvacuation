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


"""
FLOOR STUFF
"""


class FloorObject(Agent):
    def __init__(self, pos, traversable, flammable, spreads_smoke, visibility=2, model=None):
        super().__init__(pos, model)
        self.pos = pos
        self.traversable = traversable
        self.flammable = flammable
        self.spreads_smoke = spreads_smoke
        self.visibility = visibility

    def get_position(self):
        return self.pos


class Sight(FloorObject):
    def __init__(self, pos, model):
        super().__init__(pos, traversable=True, flammable=False, spreads_smoke=True, visibility=-1, model=model)

    def get_position(self):
        return self.pos


class Door(FloorObject):
    def __init__(self, pos, model):
        super().__init__(pos, traversable=True, flammable=False, spreads_smoke=True, model=model)


class FireExit(FloorObject):
    def __init__(self, pos, model):
        super().__init__(pos, traversable=True, flammable=False, spreads_smoke=False, visibility=6, model=model)


class Wall(FloorObject):
    def __init__(self, pos, model):
        super().__init__(pos, traversable=False, flammable=False, spreads_smoke=False, model=model)


class Furniture(FloorObject):
    def __init__(self, pos, model):
        super().__init__(pos, traversable=False, flammable=True, spreads_smoke=True, model=model)


"""
FIRE STUFF
"""


class Fire(FloorObject):
    """
    A fire agent

    Attributes:
        ...
    """

    def __init__(self, pos, model):
        super().__init__(pos, traversable=False, flammable=False, spreads_smoke=True, visibility=20, model=model)
        self.smoke_radius = 1

    def step(self):
        neighborhood = self.model.grid.get_neighborhood(self.pos, moore=False, include_center=False, radius=self.smoke_radius)

        for neighbor in neighborhood:
            place_smoke = True
            place_fire = True
            contents = self.model.grid.get_cell_list_contents(neighbor)

            if contents:
                for agent in contents:
                    if not agent.flammable:
                        place_fire = False
                    if not agent.spreads_smoke:
                        place_smoke = False
                    if place_fire and place_smoke:
                        break

            else:
                place_fire = False

            if place_fire:
                fire = Fire(neighbor, self.model)
                self.model.grid.place_agent(fire, neighbor)
                self.model.schedule.add(fire)
            if place_smoke:
                smoke = Smoke(neighbor, self.model)
                self.model.grid.place_agent(smoke, neighbor)
                self.model.schedule.add(smoke)

    def get_position(self):
        return self.pos


class Smoke(FloorObject):
    """
    A smoke agent

    Attributes:
        ...
    """

    def __init__(self, pos, model):
        super().__init__(pos, traversable=True, flammable=False, spreads_smoke=False, model=model)
        self.smoke_radius = 1
        self.spread_rate = 1  # The increment per step to increase self.spread by
        self.spread_threshold = 1
        self.spread = 0  # When equal or greater than spread_threshold, the smoke will spread to its neighbors

    def step(self):
        if self.spread >= 1:
            smoke_neighborhood = self.model.grid.get_neighborhood(self.pos, moore=False, include_center=False, radius=self.smoke_radius)
            for neighbor in smoke_neighborhood:
                place_smoke = True
                contents = self.model.grid.get_cell_list_contents(neighbor)
                for agent in contents:
                    if not agent.spreads_smoke:
                        place_smoke = False
                        break

                if place_smoke:
                    smoke = Smoke(neighbor, self.model)
                    self.model.grid.place_agent(smoke, neighbor)
                    self.model.schedule.add(smoke)

        if self.spread >= self.spread_threshold:
            self.spread_rate = 0
        else:
            self.spread += self.spread_rate

    def get_position(self):
        return self.pos


class DeadHuman(FloorObject):
    def __init__(self, pos, model):
        super().__init__(pos, traversable=True, flammable=True, spreads_smoke=True, model=model)


class Human(Agent):
    """
    A human agent, which will attempt to escape from the grid.

    Attributes:
        ID: Unique identifier of the Agent
        Position (x,y): Position of the agent on the Grid
        Health: Health of the agent (between 0 and 1)
        ...
    """

    def __init__(self, pos, health, speed, vision, collaboration, knowledge, nervousness, role, experience, believes_alarm, model):
        super().__init__(pos, model)
        self.traversable = False

        self.flammable = True
        self.spreads_smoke = True

        self.pos = pos
        self.visibility = 2
        self.health = health
        self.mobility = 1
        self.shock = 0
        self.speed = speed
        self.vision = vision
        self.collaboration = collaboration
        self.collaboration_count = 0
        self.knowledge = knowledge
        self.nervousness = nervousness
        self.role = role
        self.experience = experience
        self.believes_alarm = believes_alarm  # Boolean stating whether or not the agent believes the alarm is a real fire
        self.escaped = False
        self.planned_target = (None, None)  # The location (agent, (x, y)) the agent is planning to move to
        self.planned_action = None  # An action the agent intends to do when they reach their planned target {"carry", "morale"}

        self.visible_tiles = None

        # An empty set representing what the agent knows of the floor plan
        self.known_tiles = set()

        # A set representing where the agent has between
        self.visited_tiles = {self.pos}

    # A strange implementation of ray-casting, using Bresenham's Line Algorithm, which takes into account smoke and visibility of objects
    def get_visible_tiles(self):
        neighborhood = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=True, radius=self.vision)
        visible_neighborhood = set()

        for pos in neighborhood:
            blocked = False
            try:
                smoke_count = 0  # The number of smoke tiles encountered in the path so far
                path = get_line(self.pos, pos)
                for tile in path:
                    contents = self.model.grid.get_cell_list_contents(tile)
                    visible_contents = []
                    for obj in contents:
                        if isinstance(obj, Wall):  # We hit a wall, reject rest of path and move to next
                            blocked = True
                            break
                        elif isinstance(obj, Smoke):  # We hit a smoke tile, increase the counter
                            smoke_count += 1

                        # If the object has a visibility score greater than the smoke encountered in the path, it's visible
                        if obj.visibility and obj.visibility > smoke_count:
                            visible_contents.append(obj)

                    if blocked:
                        break
                    else:
                        # If a wall didn't block the way, add the visible agents at this location
                        visible_neighborhood.add((tuple(visible_contents), tile))

            except Exception as e:
                print(e)

        if self.model.visualise_vision:
            for _, tile in visible_neighborhood:
                if self.model.grid.is_cell_empty(tile):
                    sight_object = Sight(tile, self.model)
                    self.model.grid.place_agent(sight_object, tile)

        return list(visible_neighborhood)

    def get_random_target(self, allow_visited=True):
        graph_nodes = self.model.graph.nodes()

        tiles = self.known_tiles

        # If we are excluding visited tiles, remove the visited_tiles set from the available tiles
        if not allow_visited:
            tiles -= self.visited_tiles

        while not self.planned_target[1]:
            target_contents, target_pos = random.choice(list(tiles))
            if target_pos in graph_nodes and target_pos != self.pos:
                self.planned_target = (target_contents, target_pos)

    def attempt_exit_plan(self):
        self.planned_target = (None, None)
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
                        self.planned_target = (exit, exit_pos)

            else:
                self.planned_target = fire_exits.pop()

            # print("Agent found a fire escape!", self.planned_target)
        else:  # If there's a fire and no fire-escape in sight, try to head for an unvisited door, if no door in sight, move randomly (for now)
            for contents, pos in self.visible_tiles:
                for agent in contents:
                    if isinstance(agent, Door) and pos not in self.visited_tiles:
                        # print("FOUND NEW DOOR IN SIGHT INSTEAD")
                        self.planned_target = (agent, pos)
                        break
                    elif self.planned_target[1]:
                        break

            # Still didn't find a planned_target, so get a random unvisited target
            if not self.planned_target[1]:
                self.get_random_target(allow_visited=False)

    def get_panic_score(self):
        health_component = (1 / np.exp(self.health / self.nervousness))
        experience_component = (1 / np.exp(self.experience / self.nervousness))
        panic_score = (health_component + experience_component + self.shock) / 3  # Calculate the mean of the components

        # print("Panic score:", panic_score, "Health Score:", health_component, "Experience Score:", experience_component, "Shock score:", self.shock)

        return panic_score

    def die(self):
        pos = self.pos  # Store the agent's position of death so we can remove them and place a DeadHuman
        self.model.grid.remove_agent(self)
        dead_self = DeadHuman(pos, self.model)
        self.model.grid.place_agent(dead_self, pos)
        print("Agent died at", pos)

    def health_mobility_rules(self):
        moore_neighborhood = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=True, radius=1)
        contents = self.model.grid.get_cell_list_contents(moore_neighborhood)
        for agent in contents:
            if isinstance(agent, Fire):
                self.health -= 0.2
                self.speed -= 2
            elif isinstance(agent, Smoke):
                self.health -= 0.01

                # Start to slow the agent when they drop below 50% health
                if self.health < 0.5:
                    self.speed -= 1

        # Prevent health and speed from going below 0
        if self.health < 0:
            self.health = 0
        if self.speed < 0:
            self.speed = 0

        if self.health == 0:
            self.die()
        elif self.speed == 0:
            self.mobility = 0

    def panic_rules(self):
        DEFAULT_SHOCK = -0.1
        shock_modifier = DEFAULT_SHOCK  # Shock will decrease by this amount if no new shock is added
        for agents, pos in self.visible_tiles:
            for agent in agents:
                if isinstance(agent, Fire):
                    shock_modifier += 0.6
                if isinstance(agent, Smoke):
                    shock_modifier += 0.3
                if isinstance(agent, DeadHuman):
                    shock_modifier += 1.1

        # If the agent's shock value increased and they didn't believe the alarm before, they now do believe it
        if not self.believes_alarm and shock_modifier != DEFAULT_SHOCK:
            print("Agent now believes the fire is real!")
            self.believes_alarm = True

        self.shock += shock_modifier

        # Keep the shock value between 0 and 1
        if self.shock > 1:
            self.shock = 1
        elif self.shock < 0:
            self.shock = 0

        panic_score = self.get_panic_score()

        if panic_score >= 0.8 and self.mobility == 1:
            print("Agent is panicking! Score:", panic_score, "Shock:", self.shock)
            self.mobility = 2
        elif panic_score < 0.8 and self.mobility == 2:
            print("Agent stopped panicking! Score:", panic_score, "Shock:", self.shock)
            self.mobility = 1

    def learn_environment(self):
        if self.knowledge < 1:  # If there is still something to learn
            new_tiles = 0
            for agent, pos in self.visible_tiles:
                if (agent, pos) not in self.known_tiles:
                    self.known_tiles.add((agent, pos))
                    new_tiles += 1

            # update the knowledge Attribute accordingly
            total_tiles = self.model.grid.width * self.model.grid.height
            new_knowledge_percentage = new_tiles / total_tiles
            self.knowledge = self.knowledge + new_knowledge_percentage
            # print("Current knowledge:", self.knowledge)

    def get_collaboration_cost(self):
        panic_score = self.get_panic_score()
        collaboration_cost = 1 / np.exp(self.collaboration_count / self.collaboration_factor) * panic_score
        print("Collaboration cost:", collaboration_cost, "Panic score:", panic_score)
        return collaboration_cost

    def test_collaboration(self):
        collaboration_cost = self.get_collaboration_cost()

        rand = random.random()
        print(rand)
        if rand < collaboration_cost:
            return True
        else:
            return False

    def check_for_collaboration(self):
        if not self.planned_action:  # If an agent already has a planned action, they aren't able to collaborate again until it's done
            for agent, location in self.visible_tiles:
                if isinstance(agent, Human):
                    print("Mobility:", agent.get_mobility())
                    if agent.get_mobility() == 0:
                        # Physical collaboration
                        self.planned_target = (agent, location)  # Plan to move toward the target
                        self.planned_action = "carry"  # Plan to carry the agent
                        self.collaboration_count += 1
                        print("Agent planned physical collaboration at", location)
                        break
                    elif agent.get_mobility() == 2:
                        # Morale collaboration
                        self.planned_target = (agent, location)  # Plan to move toward the target
                        self.planned_action = "morale"  # Plan to carry the agent
                        self.collaboration_count += 1
                        print("Agent planned physical collaboration at", location)
                        break
                elif isinstance(agent, FireExit):
                    # Verbal collaboration
                    print("Agent verbally collaborated")
                    self.collaboration_count += 1

    def get_next_location(self, path):
        try:
            length = len(path)
            if length <= self.speed:
                next_location = path[length - 1]
            else:
                next_location = path[self.speed]

            next_path = []
            for location in path:
                next_path.append(location)
                if location == next_location:
                    break

            return (next_location, next_path)
        except Exception as e:
            print("Failed to get next location:", e, "\nPath:", path, length, "Speed:", self.speed)
            sys.exit(1)

    def get_path(self, graph, target):
        try:
            if target in self.visible_tiles:
                # print("PLANNED POS IS VISIBLE")
                return nx.shortest_path(graph, self.pos, target)
            else:
                # print("PLANNED POS IS NOT VISIBLE")
                # TODO: Replace with something more humanly (less efficient)
                return nx.shortest_path(graph, self.pos, target)
        except nx.exception.NodeNotFound as e:
            graph_nodes = graph.nodes()

            if target not in graph_nodes:
                contents = self.model.grid.get_cell_list_contents(target)
                print("Target node not found!", target, contents)
                return None
            elif self.pos not in graph_nodes:
                contents = self.model.grid.get_cell_list_contents(self.pos)
                print("Current position not found!", self.pos, contents)
                sys.exit(1)
            else:
                print(e)
                sys.exit(1)
        except nx.exception.NetworkXNoPath as e:
            # print("No path between nodes!", self.pos, target)
            return None

    def location_is_traversable(self, pos):
        if not self.model.grid.is_cell_empty(pos):
            contents = self.model.grid.get_cell_list_contents(pos)
            for agent in contents:
                if not agent.traversable:
                    return False

        return True

    def check_retreat(self, next_path, next_location):
        # Get the contents of any visible locations in the next path
        visible_path = []
        for location in next_path:
            if location in self.visible_tiles:
                visible_path.append(location)

        visible_contents = self.model.grid.get_cell_list_contents(visible_path)
        for agent in visible_contents:
            if (isinstance(agent, Smoke) and not self.planned_action) or isinstance(agent, Fire):
                # There's a danger in the visible path, so try and retreat in the opposite direction
                # Retreat if there's fire, or smoke (and no collaboration attempt)
                x, y = self.pos
                next_x, next_y = next_location
                diff_x = x - next_x
                diff_y = y - next_y
                retreat_location = (sum([x, diff_x]), sum([y, diff_y]))

                if self.model.grid.out_of_bounds(retreat_location):
                    print("retreat location out of bounds...")

                print("Agent retreating from fire/smoke")
                self.planned_target = (None, retreat_location)

                return True

    def move_toward_target(self):
        next_location = None
        graph = deepcopy(self.model.graph)

        while self.planned_target[1] and not next_location:
            path = self.get_path(graph, self.planned_target[1])

            if path:
                next_location, next_path = self.get_next_location(path)  # The final location and path traversed to said location in this step

                if self.check_retreat(next_path, next_location):
                    # We are retreating and therefore need to try a totally new path, so continue from the start of the loop
                    continue

                # Test the next location to see if we can move there
                if self.location_is_traversable(next_location):  # Move normally
                    self.previous_pos = self.pos
                    self.model.grid.move_agent(self, next_location)
                    self.visited_tiles.add(next_location)
                else:  # We want to move here but it's blocked, so remove this node from the traversable graph
                    # Remove the next location from the temporary graph so we can try pathing again without it
                    graph.remove_node(next_location)

                    # Also remove planned_target if that was the next location and break out of movement
                    if next_location == self.planned_target[1]:
                        next_location = None
                        self.planned_target = (None, None)
                        break
                    else:
                        next_location = None

                if self.pos == self.planned_target[1]:
                    # The human reached their target!

                    if self.planned_action:  # If they had an action to perform when they reached the target
                        print("Do action")
                        self.planned_action = None

                    self.planned_target = (None, None)
                    break
            else:  # No path is possible, so drop the target
                self.planned_target = (None, None)
                break

    def step(self):
        if not self.escaped and self.pos:
            self.health_mobility_rules()

            if self.health > 0:
                self.visible_tiles = self.get_visible_tiles()

                self.panic_rules()

                self.learn_environment()

                planned_agent = self.planned_target[0]

                # If a fire has started and the agent believes it, attempt to plan an exit location if we haven't already
                if self.model.fire_started and self.believes_alarm and not isinstance(planned_agent, FireExit):
                    self.attempt_exit_plan()

                # Check if anything in vision can be collaborated with
                self.check_for_collaboration()

                planned_pos = self.planned_target[1]
                if not planned_pos:
                    self.get_random_target()

                if self.mobility == 0:  # Incapacitated
                    return
                elif self.mobility == 2:  # Panic
                    if random.random() < self.get_panic_score():  # Test their panic score to see if they will move randomly, or keep their original target
                        self.planned_action = None
                        self.planned_target = (None, None)
                        self.get_random_target()

                self.move_toward_target()

                # Agent reached a fire escape, proceed to exit
                if self.model.fire_started and self.pos in self.model.fire_exit_list:
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
