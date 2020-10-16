import unittest
from envs.halite.halite_env import *


class MockPositionedElement:
    def __init__(self, position):
        self.position = position


def map_to_obs(displace, item):
    return displace + [item]


class NClosestTests(unittest.TestCase):
    def test_1(self):
        board_loc = MockPositionedElement((3, 0))
        elem1 = MockPositionedElement((3, 2))
        elem2 = MockPositionedElement((1, 1))
        board_dims = (4, 4)
        self.assertEqual(observe_n_closest(board_loc, [elem1, elem2], map_to_obs, board_dims, 2),
                         [[0, 2, elem1], [-2, 1, elem2]])

    def test_2(self):
        board_loc = MockPositionedElement((3, 0))
        elem1 = MockPositionedElement((3, 2))
        elem2 = MockPositionedElement((1, 1))
        board_dims = (4, 4)
        self.assertEqual(observe_n_closest(board_loc, [elem1, elem2], map_to_obs, board_dims, 1),
                         [[0, 2, elem1]])

    def test_3(self):
        board_loc = MockPositionedElement((1, 2))
        elem1 = MockPositionedElement((3, 2))
        elem2 = MockPositionedElement((1, 1))
        board_dims = (4, 4)
        self.assertEqual(observe_n_closest(board_loc, [elem1, elem2], map_to_obs, board_dims, 1),
                         [[0, -1, elem2]])

    def test_4(self):
        board_loc = MockPositionedElement((0, 0))
        elem1 = MockPositionedElement((2, 2))
        elem2 = MockPositionedElement((3, 3)) # can wrap around board on both axes
        board_dims = (4, 4)
        self.assertEqual(observe_n_closest(board_loc, [elem1, elem2], map_to_obs, board_dims, 2),
                         [[-1, -1, elem2], [2, 2, elem1]])

    def test_5(self):
        board_loc = MockPositionedElement((0, 0))
        elem1 = MockPositionedElement((2, 2))
        elem2 = MockPositionedElement((4, 3))  # can wrap around board on both axes
        board_dims = (5, 5)
        self.assertEqual(observe_n_closest(board_loc, [elem1, elem2], map_to_obs, board_dims, 2),
                         [[-1, -2, elem2], [2, 2, elem1]])

    def test_6(self):
        self.assertEqual(displacement((5, 10), (17, 7), (21, 21)), [-9, -3])
        self.assertEqual(displacement((5, 10), (17, 13), (21, 21)), [-9, 3])


class TestHaliteTrainHelper(unittest.TestCase):
    def test_1(self):
        """
        This is kinda weird but apparently the halite <500 per cell isn't strictly enforced if
        that limit isn't possible with the starting halite ceiling: if there's going to be
        24k halite to start on the board and the board is 3x3, it just puts thousands of halite
        on random cells.
        """
        env = make("halite", configuration={"size": 5, "startingHalite": 1000})
        env.reset(2)
        board = Board(env.state[0].observation, env.configuration)
        max_halite = board.configuration.max_cell_halite
        for cell in board.cells.values():
            self.assertLessEqual(cell.halite, max_halite)

    def test_2(self):
        helper = HaliteTrainHelper(board_size=3, team_count=4, starting_halite=1000)
        board = helper.get_board()
        helper.render()
        max_halite = board.configuration.max_cell_halite
        for cell in board.cells.values():
            self.assertLessEqual(cell.halite, max_halite)
        for i in range(4):
            self.assertEqual(len(board.players[i].ships), 1)
        self.assertEqual(board.players[0].ships[0].position, (0, 2))
        self.assertEqual(board.players[1].ships[0].position, (2, 2))
        self.assertEqual(board.players[2].ships[0].position, (0, 0))
        self.assertEqual(board.players[3].ships[0].position, (2, 0))

        frames = helper.next({AgentKey(0, '0-1'): AgentAction([1, 0, 0]),
                              AgentKey(0, '0-2'): AgentAction([1, 0, 0]),
                              AgentKey(0, '0-3'): AgentAction([1, 0, 0]),
                              AgentKey(0, '0-4'): AgentAction([1, 0, 0])})
        board = helper.get_board()
        self.assertEqual(board.players[0].ships[0].position, (0, 2))
        self.assertEqual(board.players[1].ships[0].position, (2, 2))
        self.assertEqual(board.players[2].ships[0].position, (0, 0))
        self.assertEqual(board.players[3].ships[0].position, (2, 0))

        self.maxDiff = None
        start = num_observable_halite_deposits * 3 + num_observable_friendly_ships * 3
        # enemy ships for first ship
        observation = frames[AgentKey(0, '0-1')].next_obs
        self.assertEqual(observation[start + 3 * 0 + 1:start + 3 * 0 + 3], [-1.0, 0.0])
        self.assertEqual(observation[start + 3 * 1 + 1:start + 3 * 1 + 3], [0.0, 1.0])
        self.assertEqual(observation[start + 3 * 2 + 1:start + 3 * 2 + 3], [-1.0, 1.0])
        # enemy ships for second ship
        observation = frames[AgentKey(0, '0-2')].next_obs
        self.assertEqual(observation[start + 3 * 0 + 1:start + 3 * 0 + 3], [1.0, 0.0])
        self.assertEqual(observation[start + 3 * 1 + 1:start + 3 * 1 + 3], [0.0, 1.0])
        self.assertEqual(observation[start + 3 * 2 + 1:start + 3 * 2 + 3], [1.0, 1.0])

        # Try having all ships move north
        frames = helper.next({AgentKey(0, '0-1'): AgentAction([0, 1, 0]),
                              AgentKey(0, '0-2'): AgentAction([0, 1, 0]),
                              AgentKey(0, '0-3'): AgentAction([0, 1, 0]),
                              AgentKey(0, '0-4'): AgentAction([0, 1, 0])})
        board = helper.get_board()
        self.assertEqual(board.players[0].ships[0].position, (0, 0))
        self.assertEqual(board.players[1].ships[0].position, (2, 0))
        self.assertEqual(board.players[2].ships[0].position, (0, 1))
        self.assertEqual(board.players[3].ships[0].position, (2, 1))

        # enemy ships for first ship (relative displacement hasn't changed as a result of movement
        frame = frames[AgentKey(0, '0-1')]
        self.assertEqual(frame.done, False)
        observation = frame.next_obs
        start = num_observable_halite_deposits * 3 + num_observable_friendly_ships * 3
        self.assertEqual(observation[start + 3 * 0 + 1:start + 3 * 0 + 3], [-1.0, 0.0])
        self.assertEqual(observation[start + 3 * 1 + 1:start + 3 * 1 + 3], [0.0, 1.0])
        self.assertEqual(observation[start + 3 * 2 + 1:start + 3 * 2 + 3], [-1.0, 1.0])
        # enemy ships for second ship
        frame = frames[AgentKey(0, '0-2')]
        observation = frame.next_obs
        self.assertEqual(observation[start + 3 * 0 + 1:start + 3 * 0 + 3], [1.0, 0.0])
        self.assertEqual(observation[start + 3 * 1 + 1:start + 3 * 1 + 3], [0.0, 1.0])
        self.assertEqual(observation[start + 3 * 2 + 1:start + 3 * 2 + 3], [1.0, 1.0])

        for i in range(4):
            frame = frames[AgentKey(0, '0-' + str(i + 1))]
            self.assertEqual(frame.reward, 5000)
            self.assertEqual(frame.done, False)

        # Try having some ships collide
        frames = helper.next({AgentKey(0, '0-1'): AgentAction([0, 1, 0]),
                              AgentKey(0, '0-2'): AgentAction([1, 0, 0]),
                              AgentKey(0, '0-3'): AgentAction([1, 0, 0]),
                              AgentKey(0, '0-4'): AgentAction([1, 0, 0])})
        board = helper.get_board()
        self.assertEqual(len(board.players[0].ships), 0)
        self.assertEqual(board.players[1].ships[0].position, (2, 0))
        self.assertEqual(len(board.players[2].ships), 0)
        self.assertEqual(board.players[3].ships[0].position, (2, 1))

        frame = frames[AgentKey(0, '0-1')]
        self.assertEqual(frame.next_obs, None)
        self.assertEqual(frame.done, True)
        self.assertEqual(frame.reward, 0)

        frame = frames[AgentKey(0, '0-2')]
        self.assertEqual(frame.done, False)
        self.assertEqual(frame.reward, 5000)

        frame = frames[AgentKey(0, '0-3')]
        self.assertEqual(frame.done, True)
        self.assertEqual(frame.reward, 0)

        frame = frames[AgentKey(0, '0-4')]
        self.assertEqual(frame.done, False)
        self.assertEqual(frame.reward, 5000)

