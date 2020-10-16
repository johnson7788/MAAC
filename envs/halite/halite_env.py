from kaggle_environments import evaluate, make
from kaggle_environments.envs.halite.helpers import *
from typing import List, Tuple, Callable
from utils.core import *
import traceback
from envs.base_env import *
from utils.buffer import ReplayBuffer
from train import run


class HaliteKey(AgentKey):
    def __init__(self, type: int, id: str, player: int):
        super().__init__(type, id)
        self.player = player

    def __eq__(self, other):
        return (self.type, self.id, self.player) == (other.type, other.id, other.player)

    def __hash__(self):
        return hash((self.type, self.id, self.player))

    def __str__(self):
        return "HaliteKey(" + str(self.type) + ", " + str(self.id) + ", " + str(self.player) + ")"

    def __repr__(self):
        return str(self)


# Game Constants

max_cell_halite = 500
halite_regen = 1.02
max_game_turns = 400
halite_mined_per_turn = .25 # of the halite at the current cell


# Agent topologies

# Simulating partial observability for the sake of computational resources
num_observable_friendly_ships = 20
num_observable_enemy_ships = 20
num_observable_friendly_shipyards = 5
num_observable_enemy_shipyards = 5
num_observable_halite_deposits = 20

# Order matters here
agent_types = ["ship", "shipyard"]
def get_agent_index_by_name(name: str) -> int:
    return agent_types.index(name)

# Actions for each agent type
ship_actions = [None, ShipAction.NORTH, ShipAction.EAST, ShipAction.SOUTH, ShipAction.WEST, ShipAction.CONVERT]
shipyard_actions = [None, ShipyardAction.SPAWN]

"""
Currently my ship inputs are: 
* location and halite of self (X, Y, H = 3)
* location and halite of closest 40 friendly ships (40 * (X, Y, H) = 120)
* location and halite of 40 closest enemy ships
* locations of 10 closest friendly shipyards (10 * (X, Y) = 20)
* locations of 10 closest enemy shipyards
* locations and values of 40 closest halite deposits (40 * (X, Y, H) = 120)
This is a pretty substantial amount of data, so I might limit it depending on how slowly the learning is converging
on a solution.
"""
num_ship_inputs = num_observable_halite_deposits * 3 + num_observable_friendly_ships * 3 + num_observable_enemy_ships * 3 + num_observable_friendly_shipyards * 2 + num_observable_enemy_shipyards * 2
# move north, east, west, or south, hold, or convert
num_ship_outputs = len(ship_actions)

# I think the shipyard will need an identical dataset
num_shipyard_inputs = num_ship_inputs
# build or don't build
num_shipyard_outputs = len(shipyard_actions)

pad_arr = [0]


def min_dist(from_dim: int, to_dim: int, board_dim: int) -> int:
    """
    Min wrapped manhattan distance
    """
    dist = to_dim - from_dim
    if dist > board_dim / 2:
        dist -= board_dim
    elif dist < -board_dim / 2:
        dist += board_dim
    return dist


def displacement(loc_from, loc_to, board_dims):
    return [min_dist(loc_from[0], loc_to[0], board_dims[0]), min_dist(loc_from[1], loc_to[1], board_dims[1])]


def observe_n_closest(from_item, to_items, to_obs_func, board_dims, n, pad=True):
    """
    board_loc = (x, y)
    positioned_elems = [elements with position property: position = (x, y)]
    board_dims = (width, height)
    to_obs_func: Function[(displacement, item) => observation array]
    """
    tuples = [(displacement(from_item.position, to_item.position, board_dims), to_item) for to_item in to_items]
    tuples = sorted(tuples, key=lambda tup: sum([abs(val) for val in tup[0]]))[:n]
    tuples = [to_obs_func(dis, item) for dis, item in tuples]
    if pad and len(tuples) != n:
        pad_item = to_obs_func(None, None)
        for _ in range(n - len(tuples)):
            tuples.append(pad_item)
    return tuples


def get_friendly(from_item, collection):
    return [ship for ship in collection if ship.player_id == from_item.player_id]


def get_enemy(from_item, collection):
    return [ship for ship in collection if ship.player_id != from_item.player_id]


halite_visible_threshold = max_cell_halite * .1
def get_halite(from_item, board):
    return [cell for cell in board.cells.values() if cell.halite > halite_visible_threshold]


def get_friendly_ships(from_item, board):
    return get_friendly(from_item, board.ships.values())


def get_enemy_ships(from_item, board):
    return get_enemy(from_item, board.ships.values())


def get_friendly_shipyards(from_item, board):
    return get_friendly(from_item, board.shipyards.values())


def get_enemy_shipyards(from_item, board):
    return get_enemy(from_item, board.shipyards.values())


def to_halite_observation(displace, cell) -> List[float]:
    if cell is None:
        return pad_arr * 3
    return [cell.halite / max_cell_halite] + displace


def to_ship_observation(displace, ship) -> List[float]:
    if ship is None:
        return pad_arr * 3
    return [ship.halite / max_cell_halite] + displace


def to_shipyard_observation(displace, shipyard) -> List[float]:
    if shipyard is None:
        return pad_arr * 2
    return displace


def add_to_observation(final_observation, from_item, board, board_target_filter_func, item_to_observation_func, num):
    board_size = board.configuration.size
    board_dims = [board_size, board_size]
    to_items = board_target_filter_func(from_item, board)
    for observation in observe_n_closest(from_item, to_items, item_to_observation_func, board_dims, num):
        for observed_val in observation:
            final_observation.append(observed_val)


def get_ship_observation(ship: Ship, board: Board) -> List[float]:
    """
    TODO might be worth including enemy ship team since we go against more than 1 team at times
    """
    observation = []
    add_to_observation(observation, ship, board, get_halite, to_halite_observation, num_observable_halite_deposits)
    add_to_observation(observation, ship, board, get_friendly_ships, to_ship_observation, num_observable_friendly_ships)
    add_to_observation(observation, ship, board, get_enemy_ships, to_ship_observation, num_observable_enemy_ships)
    add_to_observation(observation, ship, board, get_friendly_shipyards, to_shipyard_observation, num_observable_friendly_shipyards)
    add_to_observation(observation, ship, board, get_enemy_shipyards, to_shipyard_observation, num_observable_enemy_shipyards)
    # print("Observation:", observation)
    # print("Board size = ", board.configuration.size)
    # print("Pos = ", ship.position)
    assert len(observation) == num_ship_inputs
    return observation


def get_shipyard_observation(shipyard: Shipyard, board: Board) -> List[float]:
    observation = get_ship_observation(shipyard, board) # gonna be the same for now
    assert len(observation) == num_shipyard_inputs
    return observation


def get_empty_observation_for(agent_type: int) -> List[float]:
    if agent_type == 0:
        return [0] * num_ship_inputs
    elif agent_type == 1:
        return [0] * num_shipyard_inputs


class HaliteTrainHelper(BaseEnv):
    """
    Team-agnostic dojo for training the halite environment.  Simplified to three methods: reset(), observe(), and
    act() to make training easier.  I'm trying to keep AI model code out of this though.
    """
    def __init__(self, board_size=None, team_count=2, starting_halite=1000):
        # Create Halite environment and training dojo
        config = {}
        if board_size is not None:
            config["size"] = board_size
        if starting_halite is not None:
            config["startingHalite"] = starting_halite
        self.environment = make("halite", configuration=config)
        self.team_count = team_count

    @property
    def config(self):
        return Config(reward_scale=10,
                      pol_hidden_dim=128,
                      critic_hidden_dim=128,
                      episode_length=350,
                      batch_size=300,
                      buffer_length=1500, # has to be reduced for GPU memory usage reasons
                      save_interval=70, # so we can sim the current state more often
                      games_per_update=1,
                      use_gpu=True)

    @property
    def agent_type_topologies(self) -> List[Tuple[int, int]]:
        return [(num_ship_inputs, num_ship_outputs), (num_shipyard_inputs, num_shipyard_outputs)]

    def player_reward(self, player):
        return 0 if len(player.shipyards) == 0 else player.halite

    def simulate(self, model: Callable[[Dict[AgentKey, AgentObservation]], Dict[AgentKey, AgentAction]], buffer: ReplayBuffer) -> float:
        """
        actions: for each of the active agents, an index indicating the action they selected (probably via
        softmax) should be returned (i.e. [3, 2, 3, 0])
        Return value is in the form:
        returns rewards, dones, next observations
        """

        self.environment.reset(self.team_count)
        self.board = Board(self.environment.state[0].observation, self.environment.configuration)

        sim_buffer: SimulationBuffer = SimulationBuffer()

        for turn in range(200):
            # Observe
            observations: Dict[HaliteKey, AgentObservation] = {}

            for ship in self.board.ships.values():
                observations[HaliteKey(0, ship.id, ship.player_id)] = AgentObservation(
                    get_ship_observation(ship, self.board))

            for shipyard in self.board.shipyards.values():
                observations[HaliteKey(1, shipyard.id, shipyard.player_id)] = AgentObservation(
                    get_shipyard_observation(shipyard, self.board))

            # Act
            actions: Dict[AgentKey, AgentAction] = model(observations)

            game_object_types = [list(self.board.ships.values()), list(self.board.shipyards.values())]
            for i in range(len(game_object_types)):
                for game_object in game_object_types[i]:
                    key = HaliteKey(i, game_object.id, game_object.player_id)
                    action_index = actions[key].get_action_index()
                    if i == 0:
                        game_object.next_action = ship_actions[action_index]
                    elif i == 1:
                        game_object.next_action = shipyard_actions[action_index]

            self.board = self.board.next()

            # Calculate rewards
            rewards_by_team: Dict[PlayerId, float] = {k: self.player_reward(v) for k, v in self.board.players.items()}

            rewards: Dict[HaliteKey, float] = {}
            dones: Dict[HaliteKey, bool] = {}

            for k in observations.keys():
                rewards[k] = rewards_by_team[k.player]
                if k.type == 0:
                    dones[k] = k.id not in self.board.ships
                elif k.type == 1:
                    dones[k] = k.id not in self.board.shipyards

            sim_buffer.push(observations, actions, rewards, dones)

        final_rewards_by_team: Dict[PlayerId, float] = {k: self.player_reward(v) for k, v in self.board.players.items()}

        # Push from sim buffer to actual replay buffer
        for i in range(len(sim_buffer.frames)):
            frame = sim_buffer.frames[i]
            next_frame = sim_buffer.frames[i + 1] if i < len(sim_buffer.frames) - 1 else None
            for k in frame.keys():
                if next_frame is None:
                    frame[k].next_obs = [0] * len(frame[k].obs)
                    frame[k].reward = final_rewards_by_team[k.player]
                elif k not in next_frame:
                    assert frame[k].done
                    frame[k].next_obs = [0] * len(frame[k].obs)
                    frame[k].reward = final_rewards_by_team[k.player]
                else:
                    assert not frame[k].done
                    frame[k].next_obs = next_frame[k].obs
            buffer.push({k: v.build() for k, v in frame.items()})

        return sum(final_rewards_by_team.values()) / len(final_rewards_by_team)

    def get_board(self):
        return self.board

    def set_board(self, observation):
        self.board = Board(observation, self.environment.configuration)

    def render(self):
        print(self.board)


class HaliteRunHelper:
    def __init__(self):
        self.environment = make("halite")

    def _make_decision(self, observation, configuration, obs_to_action: Callable[[Dict[HaliteKey, AgentObservation]], Dict[HaliteKey, AgentAction]]):
        print("Making decisions")
        board = Board(observation, configuration)
        current_player = board.current_player
        observations: Dict[HaliteKey, AgentObservation] = {}
        for ship in current_player.ships:
            observations[HaliteKey(0, ship.id, ship.player_id)] = AgentObservation(get_ship_observation(ship, board))
            print("Got obs for", ship.id)
        for shipyard in current_player.shipyards:
            observations[HaliteKey(1, shipyard.id, shipyard.player_id)] = AgentObservation(get_shipyard_observation(shipyard, board))
        print("Getting actions")
        try:
            actions: Dict[HaliteKey, AgentAction] = obs_to_action(observations)
        except:
            traceback.print_exc()
        print("Got actions")
        for ship in current_player.ships:
            key = HaliteKey(0, ship.id, ship.player_id)
            action_index = actions[key].get_action_index()
            ship.next_action = ship_actions[action_index]
            print("Chose", ship.next_action)
        for shipyard in current_player.shipyards:
            key = HaliteKey(1, shipyard.id, shipyard.player_id)
            action_index = actions[key].get_action_index()
            shipyard.next_action = shipyard_actions[action_index]
            print("Chose", shipyard.next_action)
        return current_player.next_actions


    def simulate(self, obs_to_action: Callable[[Dict[HaliteKey, AgentObservation]], Dict[HaliteKey, AgentAction]], agent_count=2):
        self.first_turn = True
        self.environment.reset(agent_count)
        agents = []
        for i in range(agent_count):
            agents.append(lambda o, c: self._make_decision(o, c, obs_to_action))
        self.environment.run(agents)
        render_result = self.environment.render(mode="html", width=500, height=450)
        with open("rendered.html", "w") as file:
            file.write(render_result)

        import os
        os.system("google-chrome ~/Desktop/MAAC/rendered.html")

class HaliteEnvTrain(unittest.TestCase):
    def test_env(self):
        run(HaliteTrainHelper())
