from mesa.space import Coordinate
import networkx as nx
import numpy as np
import sys
import random
from enum import Enum

from mesa import Agent

from fire_evacuation.utils import get_random_id


def get_line(start, end):
    """
    Implementaiton of Bresenham's Line Algorithm
    Returns a list of tuple coordinates from starting tuple to end tuple (and including them)
    """
    # Break down start and end tuples
    x1, y1 = start
    x2, y2 = end

    # Calculate differences
    diff_x = x2 - x1
    diff_y = y2 - y1

    # Check if the line is steep
    line_is_steep = abs(diff_y) > abs(diff_x)

    # If the line is steep, rotate it
    if line_is_steep:
        # Swap x and y values for each pair
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    # If the start point is further along the x-axis than the end point, swap start and end
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True

    # Calculate the differences again
    diff_x = x2 - x1
    diff_y = y2 - y1

    # Calculate the error margin
    error_margin = int(diff_x / 2.0)
    step_y = 1 if y1 < y2 else -1

    # Iterate over the bounding box, generating coordinates between the start and end coordinates
    y = y1
    path = []

    for x in range(x1, x2 + 1):
        coord = (
            (y, x) if line_is_steep else (x, y)
        )  # Get a coordinate according to if x and y values were swapped
        path.append(coord)  # Add it to our path
        error_margin -= abs(
            diff_y
        )  # Deduct the absolute difference of y values from our error_margin

        if (
            error_margin < 0
        ):  # When the error margin drops below zero, increase y by the step and the error_margin by the x difference
            y += step_y
            error_margin += diff_x

    # The the start and end were swapped, reverse the path
    if swapped:
        path.reverse()

    return path


"""
FLOOR STUFF
"""


class FloorObject(Agent):
    def __init__(
        self,
        pos: Coordinate,
        traversable: bool,
        flammable: bool,
        spreads_smoke: bool,
        visibility: int = 2,
        model=None,
    ):
        rand_id = get_random_id()
        super().__init__(rand_id, model)
        self.pos = pos
        self.traversable = traversable
        self.flammable = flammable
        self.spreads_smoke = spreads_smoke
        self.visibility = visibility

    def get_position(self):
        return self.pos


class Sight(FloorObject):
    def __init__(self, pos, model):
        super().__init__(
            pos, traversable=True, flammable=False, spreads_smoke=True, visibility=-1, model=model
        )

    def get_position(self):
        return self.pos


class Door(FloorObject):
    def __init__(self, pos, model):
        super().__init__(pos, traversable=True, flammable=False, spreads_smoke=True, model=model)


class FireExit(FloorObject):
    def __init__(self, pos, model):
        super().__init__(
            pos, traversable=True, flammable=False, spreads_smoke=False, visibility=6, model=model
        )


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
        super().__init__(
            pos,
            traversable=False,
            flammable=False,
            spreads_smoke=True,
            visibility=20,
            model=model,
        )
        self.smoke_radius = 1

    def step(self):
        neighborhood = self.model.grid.get_neighborhood(
            self.pos, moore=False, include_center=False, radius=self.smoke_radius
        )

        for neighbor_pos in neighborhood:
            place_smoke = False
            place_fire = False
            contents = self.model.grid.get_cell_list_contents(neighbor_pos)

            if contents:
                for agent in contents:
                    if agent.flammable:
                        place_fire = True
                    if agent.spreads_smoke:
                        place_smoke = True
                    if place_fire and place_smoke:
                        break

            if place_fire:
                fire = Fire(neighbor_pos, self.model)
                self.model.schedule.add(fire)
                self.model.grid.place_agent(fire, neighbor_pos)
            if place_smoke:
                smoke = Smoke(neighbor_pos, self.model)
                self.model.schedule.add(smoke)
                self.model.grid.place_agent(smoke, neighbor_pos)

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
            smoke_neighborhood = self.model.grid.get_neighborhood(
                self.pos, moore=False, include_center=False, radius=self.smoke_radius
            )
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

    class Mobility(Enum):
        INCAPACITATED = 0
        NORMAL = 1
        PANIC = 2

    class Status(Enum):
        DEAD = 0
        ALIVE = 1
        ESCAPED = 2

    class Action(Enum):
        PHYSICAL_SUPPORT = 0
        MORALE_SUPPORT = 1
        VERBAL_SUPPORT = 2
        RETREAT = 3

    MIN_HEALTH = 0
    MAX_HEALTH = 1

    MIN_EXPERIENCE = 1
    MAX_EXPERIENCE = 10

    MIN_SPEED = 0
    MAX_SPEED = 2

    MIN_KNOWLEDGE = 0
    MAX_KNOWLEDGE = 1

    MAX_SHOCK = 1
    MIN_SHOCK = 0
    # Shock modifiers when encountering certain objects per object, per step
    DEFAULT_SHOCK_MODIFIER = -0.1  # The default amount the shock value will change per step
    SHOCK_MODIFIER_DEAD_HUMAN = 1.0
    SHOCK_MODIFIER_FIRE = 0.2
    SHOCK_MODIFIER_SMOKE = 0.05
    SHOCK_MODIFIER_AFFECTED_HUMAN = 0.1

    PANIC_THRESHOLD = (
        0.8  # The value the panic score must reach for an agent to start panic behaviour
    )

    HEALTH_MODIFIER_FIRE = 0.2
    HEALTH_MODIFIER_SMOKE = 0.005

    SPEED_MODIFIER_FIRE = 2
    SPEED_MODIFIER_SMOKE = 1

    SLOWDOWN_THRESHOLD = (
        0.5  # When the health value drops below this value, the agent will being to slow down
    )

    def __init__(
        self,
        pos,
        health,
        speed,
        vision,
        collaborates,
        nervousness,
        experience,
        believes_alarm,
        model,
    ):
        rand_id = get_random_id()
        super().__init__(rand_id, model)
        self.traversable = False

        self.flammable = True
        self.spreads_smoke = True

        self.pos = pos
        self.visibility = 2
        self.health = health
        self.mobility = Human.Mobility.NORMAL
        self.shock = self.MIN_SHOCK
        self.speed = speed
        self.vision = vision

        self.collaborates = (
            collaborates  # Boolean specifying whether this agent will attempt collaboration
        )
        self.verbal_collaboration_count = 0
        self.morale_collaboration_count = 0
        self.physical_collaboration_count = 0

        self.morale_boost = False
        self.carried = False
        self.carrying = None

        self.knowledge = self.MIN_KNOWLEDGE
        self.nervousness = nervousness
        self.experience = experience
        self.believes_alarm = believes_alarm  # Boolean stating whether or not the agent believes the alarm is a real fire
        self.escaped = False
        self.planned_target = (
            None,
            None,
        )  # The location (agent, (x, y)) the agent is planning to move to
        self.planned_action = None  # An action the agent intends to do when they reach their planned target {"carry", "morale"}

        self.visible_tiles = None

        # An empty set representing what the agent knows of the floor plan
        self.known_tiles = set()

        # A set representing where the agent has between
        self.visited_tiles = {self.pos}

    def update_sight_tiles(self, visible_neighborhood):
        if self.visible_tiles:
            # Remove old vision tiles
            for tile in self.visible_tiles:
                contents = self.model.grid.get_cell_list_contents(tile[1])
                for agent in contents:
                    if isinstance(agent, Sight):
                        self.model.grid.remove_agent(agent)

        # Add new vision tiles
        for contents, tile in visible_neighborhood:
            if (
                self.model.grid.is_cell_empty(tile) or contents
            ):  # Don't place if the tile has contents but the agent can't see it
                sight_object = Sight(tile, self.model)
                self.model.grid.place_agent(sight_object, tile)

    # A strange implementation of ray-casting, using Bresenham's Line Algorithm, which takes into account smoke and visibility of objects
    def get_visible_tiles(self):
        neighborhood = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=True, radius=self.vision
        )
        visible_neighborhood = set()
        checked_tiles = (
            set()
        )  # A set of already checked tiles, for avoiding repetition and thus increased efficiency

        for pos in reversed(
            neighborhood
        ):  # Reverse the neighborhood so we start from the furthest locations and work our way inwards
            if pos not in checked_tiles:
                blocked = False
                try:
                    smoke_count = 0  # The number of smoke tiles encountered in the path so far
                    path = get_line(self.pos, pos)

                    for i, tile in enumerate(path):
                        contents = self.model.grid.get_cell_list_contents(tile)
                        visible_contents = []
                        for obj in contents:
                            if isinstance(
                                obj, Wall
                            ):  # We hit a wall, reject rest of path and move to next
                                blocked = True
                                break
                            elif isinstance(
                                obj, Smoke
                            ):  # We hit a smoke tile, increase the counter
                                smoke_count += 1

                            # If the object has a visibility score greater than the smoke encountered in the path, it's visible
                            if obj.visibility and obj.visibility > smoke_count:
                                visible_contents.append(obj)

                        if blocked:
                            checked_tiles.update(
                                path[i:]
                            )  # Add the rest of the path to checked tiles, since we now know they are not visible
                            break
                        else:
                            # If a wall didn't block the way, add the visible agents at this location
                            checked_tiles.add(
                                tile
                            )  # Add the tile to checked tiles so we don't check it again
                            visible_neighborhood.add((tuple(visible_contents), tile))

                except Exception as e:
                    print(e)

        if self.model.visualise_vision:
            self.update_sight_tiles(visible_neighborhood)

        return list(visible_neighborhood)

    def get_random_target(self, allow_visited=True):
        graph_nodes = self.model.graph.nodes()

        tiles = self.known_tiles

        # If we are excluding visited tiles, remove the visited_tiles set from the available tiles
        if not allow_visited:
            tiles -= self.visited_tiles

        while not self.planned_target[1]:
            _, target_pos = random.choice(list(tiles))
            if target_pos in graph_nodes and target_pos != self.pos:
                self.planned_target = (None, target_pos)

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
                    length = len(
                        get_line(self.pos, exit_pos)
                    )  # Let's use Bresenham's to find the 'closest' exit
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
        health_component = 1 / np.exp(self.health / self.nervousness)
        experience_component = 1 / np.exp(self.experience / self.nervousness)
        panic_score = (
            health_component + experience_component + self.shock
        ) / 3  # Calculate the mean of the components

        # print("Panic score:", panic_score, "Health Score:", health_component, "Experience Score:", experience_component, "Shock score:", self.shock)

        return panic_score

    def die(self):
        pos = (
            self.pos
        )  # Store the agent's position of death so we can remove them and place a DeadHuman
        self.model.grid.remove_agent(self)
        dead_self = DeadHuman(pos, self.model)
        self.model.grid.place_agent(dead_self, pos)
        print("Agent died at", pos)

    def health_mobility_rules(self):
        moore_neighborhood = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=True, radius=1
        )
        contents = self.model.grid.get_cell_list_contents(moore_neighborhood)

        for agent in contents:
            if isinstance(agent, Fire):
                self.health -= self.HEALTH_MODIFIER_FIRE
                self.speed -= self.SPEED_MODIFIER_FIRE
            elif isinstance(agent, Smoke):
                self.health -= self.HEALTH_MODIFIER_SMOKE

                # Start to slow the agent when they drop below 50% health
                if self.health < self.SLOWDOWN_THRESHOLD:
                    self.speed -= self.SPEED_MODIFIER_SMOKE

        # Prevent health and speed from going below 0
        if self.health < self.MIN_HEALTH:
            self.health = self.MIN_HEALTH
        if self.speed < self.MIN_SPEED:
            self.speed = self.MIN_SPEED

        if self.health == self.MIN_HEALTH:
            self.stop_carrying()
            self.die()
        elif self.speed == self.MIN_SPEED:
            self.stop_carrying()
            self.mobility = Human.Mobility.INCAPACITATED

    def panic_rules(self):
        if self.morale_boost:  # If the agent recieved a morale boost, they will not panic again
            return

        shock_modifier = (
            self.DEFAULT_SHOCK_MODIFIER
        )  # Shock will decrease by this amount if no new shock is added
        for agents, pos in self.visible_tiles:
            for agent in agents:
                if isinstance(agent, Fire):
                    shock_modifier += self.SHOCK_MODIFIER_FIRE - self.DEFAULT_SHOCK_MODIFIER
                if isinstance(agent, Smoke):
                    shock_modifier += self.SHOCK_MODIFIER_SMOKE - self.DEFAULT_SHOCK_MODIFIER
                if isinstance(agent, DeadHuman):
                    shock_modifier += self.SHOCK_MODIFIER_DEAD_HUMAN - self.DEFAULT_SHOCK_MODIFIER
                if isinstance(agent, Human) and agent.get_mobility() != Human.Mobility.NORMAL:
                    shock_modifier += (
                        self.SHOCK_MODIFIER_AFFECTED_HUMAN - self.DEFAULT_SHOCK_MODIFIER
                    )

        # If the agent's shock value increased and they didn't believe the alarm before, they now do believe it
        if not self.believes_alarm and shock_modifier != self.DEFAULT_SHOCK_MODIFIER:
            print("Agent now believes the fire is real!")
            self.believes_alarm = True

        self.shock += shock_modifier

        # Keep the shock value between 0 and 1
        if self.shock > self.MAX_SHOCK:
            self.shock = self.MAX_SHOCK
        elif self.shock < self.MIN_SHOCK:
            self.shock = self.MIN_SHOCK

        panic_score = self.get_panic_score()

        if panic_score >= self.PANIC_THRESHOLD and self.mobility == Human.Mobility.NORMAL:
            print("Agent is panicking! Score:", panic_score, "Shock:", self.shock)
            self.stop_carrying()
            self.mobility = Human.Mobility.PANIC
        elif panic_score < self.PANIC_THRESHOLD and self.mobility == Human.Mobility.PANIC:
            print("Agent stopped panicking! Score:", panic_score, "Shock:", self.shock)
            self.mobility = Human.Mobility.NORMAL

    def learn_environment(self):
        if self.knowledge < self.MAX_KNOWLEDGE:  # If there is still something to learn
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
        total_count = (
            self.verbal_collaboration_count
            + self.morale_collaboration_count
            + self.physical_collaboration_count
        )
        collaboration_component = 1 / np.exp(
            1 / (total_count + 1)
        )  # The more time this agent has collaborated, the higher the score will become
        collaboration_cost = (collaboration_component + panic_score) / 2
        # print("Collaboration cost:", collaboration_cost, "Component:", collaboration_component, "Panic component:", panic_score)

        return collaboration_cost

    def test_collaboration(self):
        collaboration_cost = self.get_collaboration_cost()

        rand = random.random()
        if (
            rand > collaboration_cost
        ):  # Collaboration if rand is GREATER than our collaboratoin_cost (Highe collaboration_cost means less likely to collaborate)
            return True
        else:
            return False

    def verbal_collaboration(self, target_agent, target_location):
        success = False
        for agents, location in self.visible_tiles:
            for agent in agents:
                if isinstance(agent, Human) and agent.get_mobility() == Human.Mobility.NORMAL:
                    agent_target, agent_action = agent.get_plan()
                    if not agent_action and not isinstance(
                        agent_target[0], FireExit
                    ):  # If the agent isn't planning an action and isn't already heading to a fire exit
                        agent.set_plan(
                            target_agent, target_location
                        )  # Set the agent's planned_target to the target location
                        agent.set_believes(
                            True
                        )  # The agent should now beleive it's real if they didn't before
                        success = True

        if success:
            print("Agent informed others of a fire exit!")
            self.verbal_collaboration_count += 1

    def check_for_collaboration(self):
        if (
            self.carrying
        ):  # If the agent is carrying someone, they are too occupied to do other collaboration
            return

        if self.test_collaboration():
            for visible_agents, location in self.visible_tiles:
                if self.planned_action:
                    break

                for agent in visible_agents:
                    if isinstance(agent, Human) and not self.planned_action:
                        if (
                            agent.get_mobility() == Human.Mobility.INCAPACITATED
                        ):  # If the agent is incapacitated, help them
                            # Physical collaboration
                            self.planned_target = (
                                agent,
                                location,
                            )  # Plan to move toward the target
                            self.planned_action = (
                                Human.Action.PHYSICAL_SUPPORT
                            )  # Plan to carry the agent
                            # print("Agent planned physical collaboration at", location)
                            break
                        elif (
                            agent.get_mobility() == Human.Mobility.PANIC
                            and not self.planned_action
                        ):
                            # Morale collaboration
                            self.planned_target = (
                                agent,
                                location,
                            )  # Plan to move toward the target
                            self.planned_action = (
                                Human.Action.MORALE_SUPPORT
                            )  # Plan to carry the agent
                            # print("Agent planned morale collaboration at", location)
                            break
                    elif isinstance(agent, FireExit):
                        # Verbal collaboration
                        self.verbal_collaboration(agent, location)

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

    def get_path(self, graph, target, include_target=True):
        try:
            if target in self.visible_tiles:  # Target is visible, so simply take the shortest path
                path = nx.shortest_path(graph, self.pos, target)
            else:  # Target is not visible, so do less efficient pathing
                # TODO: In the future this could be replaced with a more naive path algorithm
                path = nx.shortest_path(graph, self.pos, target)

                if not include_target:
                    del path[
                        -1
                    ]  # We don't want the target included in the path, so delete the last element

            return path
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

    def get_retreat_location(self, next_location):
        x, y = self.pos
        next_x, next_y = next_location
        diff_x = x - next_x
        diff_y = y - next_y

        retreat_location = (sum([x, diff_x]), sum([y, diff_y]))
        return retreat_location

    def check_retreat(self, next_path, next_location):
        # Get the contents of any visible locations in the next path
        visible_path = []
        for visible_agent, visible_pos in self.visible_tiles:
            if visible_pos in next_path:
                visible_path.append(visible_pos)

        visible_contents = self.model.grid.get_cell_list_contents(visible_path)
        for agent in visible_contents:
            if (isinstance(agent, Smoke) and not self.planned_action) or isinstance(agent, Fire):
                # There's a danger in the visible path, so try and retreat in the opposite direction
                # Retreat if there's fire, or smoke (and no collaboration attempt)
                retreat_location = self.get_retreat_location(next_location)

                # Check if retreat location is out of bounds
                if not self.model.grid.out_of_bounds(retreat_location):
                    # Check if the retreat location is also smoke, if so, we are surrounded by smoke, so move randomly
                    contents = self.model.grid.get_cell_list_contents(retreat_location)
                    for agent in contents:
                        if isinstance(agent, Smoke) or isinstance(agent, Fire):
                            self.get_random_target()
                            print("Agent surrounded by smoke and moving randomly")
                            retreat_location = None
                            break

                    if retreat_location:
                        print("Agent retreating opposite to fire/smoke")
                        self.planned_target = (None, retreat_location)
                else:
                    self.get_random_target()  # Since our retreat is out of bounds, just go to a random location

                self.planned_action = Human.Action.RETREAT
                return True

    def update_target(self):
        # If there was a target agent, check if target has moved or still exists
        planned_agent = self.planned_target[0]
        if planned_agent:
            current_pos = planned_agent.get_position()
            if current_pos and current_pos != self.planned_target[1]:  # Agent has moved
                self.planned_target = (planned_agent, current_pos)
                # print("Target agent moved. Updating current position:", self.planned_target)
            elif not current_pos:  # Agent no longer exists
                # print("Target agent no longer exists. Dropping.", self.planned_target, current_pos)
                self.planned_target = (None, None)
                self.planned_action = None

    def update_action(self):
        planned_agent, _ = self.planned_target

        if planned_agent:
            # Agent had planned morale collaboration, but the agent is no longer panicking or no longer alive, so drop it.
            if self.planned_action == Human.Action.MORALE_SUPPORT and (
                planned_agent.get_mobility() != Human.Mobility.PANIC
                or not planned_agent.get_status() == Human.Status.ALIVE
            ):
                # print("Target agent no longer panicking. Dropping action.")
                self.planned_target = (None, None)
                self.planned_action = None
            # Agent had planned physical collaboration, but the agent is no longer incapacitated or has already been carried or is not alive, so drop it.
        elif self.planned_action == Human.Action.PHYSICAL_SUPPORT and (
            (planned_agent.get_mobility() != Human.Mobility.INCAPACITATED)
            or planned_agent.is_carried()
            or not planned_agent.get_status() == Human.Status.ALIVE
        ):
            self.planned_target = (None, None)
            self.planned_action = None
        elif self.planned_action == Human.Action.RETREAT:
            return
        else:  # Can no longer perform the action
            self.planned_target = (None, None)
            self.planned_action = None

    def perform_action(self):
        agent, _ = self.planned_target

        if self.planned_action == Human.Action.PHYSICAL_SUPPORT:
            if not agent.is_carried():
                self.carrying = agent
                agent.set_carried(True)
                self.physical_collaboration_count += 1
                print("Agent started carrying another agent")
        elif self.planned_action == Human.Action.MORALE_SUPPORT:
            # Attempt to give the agent a permanent morale boost according to your experience score
            if agent.attempt_morale_boost(self.experience):
                print("Morale boost succeeded")
            else:
                print("Morale boost failed")

            self.morale_collaboration_count += 1

        self.planned_action = None

    def move_toward_target(self):
        next_location = None
        pruned_edges = set()

        self.update_target()  # Get the latest location of a target, if it still exists
        if self.planned_action:  # And if there's an action, check if it's still possible
            self.update_action()

        while self.planned_target[1] and not next_location:
            if self.location_is_traversable(self.planned_target[1]):  # Target is traversable
                path = self.get_path(self.model.graph, self.planned_target[1])
            else:  # Target is not traversable (e.g. we are going to another Human), so don't include target in the path
                path = self.get_path(
                    self.model.graph, self.planned_target[1], include_target=False
                )

            if path:
                next_location, next_path = self.get_next_location(
                    path
                )  # The final location and path traversed to said location in this step

                if self.check_retreat(next_path, next_location):
                    # We are retreating and therefore need to try a totally new path, so continue from the start of the loop
                    continue

                # Test the next location to see if we can move there
                if self.location_is_traversable(next_location):  # Move normally
                    self.previous_pos = self.pos
                    self.model.grid.move_agent(self, next_location)
                    self.visited_tiles.add(next_location)

                    if self.carrying:
                        agent = self.carrying
                        if agent.get_status() != Human.Status.DEAD:
                            try:
                                self.model.grid.move_agent(self.carrying, self.pos)
                            except Exception as e:
                                agent = self.carrying
                                print(
                                    "Failed to move carried agent",
                                    agent,
                                    agent.get_mobility(),
                                    agent.get_status(),
                                    agent.get_health(),
                                    agent.get_position(),
                                    self.pos,
                                )
                                sys.exit(1)
                        else:  # Agent is dead, so we can't carry them any more
                            self.stop_carrying()

                elif self.pos == path[-1]:
                    # The human reached their target!

                    if (
                        self.planned_action
                    ):  # If they had an action to perform when they reached the target
                        self.perform_action()

                    self.planned_target = (None, None)
                    self.planned_action = None
                    break

                else:  # We want to move here but it's blocked, so remove this node from the traversable graph
                    # Remove the next location from the temporary graph so we can try pathing again without it
                    edges = self.model.graph.edges(next_location)
                    pruned_edges.update(edges)
                    self.model.graph.remove_node(next_location)

                    # Reset planned_target if the next location was the end of the path
                    if next_location == path[-1]:
                        next_location = None
                        self.planned_target = (None, None)
                        self.planned_action = None
                        break
                    else:
                        next_location = None

            else:  # No path is possible, so drop the target
                self.planned_target = (None, None)
                self.planned_action = None
                break

        if pruned_edges:
            # Add back the edges we removed when removing any non-traversable nodes from the global graph, because they may be traversable again next step
            self.model.graph.add_edges_from(list(pruned_edges))

    def step(self):
        if not self.escaped and self.pos:
            self.health_mobility_rules()

            if (
                self.mobility == Human.Mobility.INCAPACITATED or not self.pos
            ):  # Incapacitated or died, so return already
                return

            self.visible_tiles = self.get_visible_tiles()

            self.panic_rules()

            self.learn_environment()

            planned_agent = self.planned_target[0]

            # If a fire has started and the agent believes it, attempt to plan an exit location if we haven't already and we aren't performing an action
            if self.model.fire_started and self.believes_alarm:
                if not isinstance(planned_agent, FireExit) and not self.planned_action:
                    self.attempt_exit_plan()

                # Check if anything in vision can be collaborated with, if the agent has normal mobility
                if self.mobility == Human.Mobility.NORMAL and self.collaborates:
                    self.check_for_collaboration()

            planned_pos = self.planned_target[1]
            if not planned_pos:
                self.get_random_target()
            elif self.mobility == Human.Mobility.PANIC:  # Panic
                panic_score = self.get_panic_score()

                if (
                    panic_score > 0.9 and random.random() < panic_score
                ):  # If they have above 90% panic score, test the score to see if they faint
                    print("Agent fainted!")
                    self.mobility = Human.Mobility.INCAPACITATED
                    return
                if (
                    random.random() < panic_score
                ):  # Test their panic score to see if they will move randomly, or keep their original target
                    print("Agent moving randomly in panic!")
                    self.planned_action = None
                    self.planned_target = (None, None)
                    self.get_random_target()

            self.move_toward_target()

            # Agent reached a fire escape, proceed to exit
            if self.model.fire_started and self.pos in self.model.fire_exits.values():
                if self.carrying:
                    carried_agent = self.carrying
                    carried_agent.escaped = True
                    self.model.grid.remove_agent(carried_agent)

                self.escaped = True
                self.model.grid.remove_agent(self)

    def get_status(self):
        if self.health > self.MIN_HEALTH and not self.escaped:
            return Human.Status.ALIVE
        elif self.health <= self.MIN_HEALTH and not self.escaped:
            return Human.Status.DEAD
        elif self.escaped:
            return Human.Status.ESCAPED

        return None

    def get_speed(self):
        return self.speed

    def get_mobility(self):
        return self.mobility

    def get_health(self):
        return self.health

    def get_position(self):
        return self.pos

    def get_plan(self):
        return (self.planned_target, self.planned_action)

    def set_plan(self, agent, location):
        self.planned_action = None
        self.planned_target = (agent, location)

    def set_believes(self, value: bool):
        if value and not self.believes_alarm:
            print("Agent told to believe the alarm!")

        self.believes_alarm = value

    def attempt_morale_boost(self, experience: int):
        rand = random.random()
        if rand < (experience / self.MAX_EXPERIENCE):
            self.morale_boost = True
            self.mobility = Human.Mobility.NORMAL
            return True
        else:
            return False

    def stop_carrying(self):
        if self.carrying:
            carried_agent = self.carrying
            carried_agent.set_carried(False)
            self.carrying = None
            self.planned_action = None
            print("Agent stopped carrying another agent")

    def set_carried(self, value: bool):
        self.carried = value

    def is_carried(self):
        return self.carried

    def is_carrying(self):
        if self.carrying:
            return True
        else:
            return False

    def get_verbal_collaboration_count(self):
        return self.verbal_collaboration_count

    def get_morale_collaboration_count(self):
        return self.morale_collaboration_count

    def get_physical_collaboration_count(self):
        return self.physical_collaboration_count
