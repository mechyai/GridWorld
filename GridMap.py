import numpy as np
import random
import math


class GridState:
    """
    Helper class for State objects that make up the Map.
    """
    def __init__(self,
                 state_loc: tuple = None,
                 id_val: str = None,
                 access: bool = None,
                 reset: bool = None,
                 actions: int = 11110000,
                 reward: tuple = None):
        """
        Creates State object with all necessary layer properties to help build Map objects more easily

        :param state_loc: (2-tuple) (x,y) or (r,c) location of agent in Map matrix
        :param id_val: (str) 2 char State ID tag specifying state class and type
        :param access: (bool) whether (true) or not (false) agent has access to the state's location
        :param reset: (bool) whether (true) or not (false) agent will reset at the state
        :param actions: (8-bit binary word) encodes actions available to agent respective to (U,D,L,R,UL,DL,UR,DR)
        :param reward: (tuple) encoded scalar reward signal via (mean_reward, distr_var) tuple
        :param value: (int) value perception of agent
        :param set_val: (bool) whether (true) or not (false) state has been modified since initial initialization
        """
        self.loc = state_loc
        self.id = id_val
        self.access = access
        self.reset = reset
        self.actions = actions
        self.reward = reward
        self.value = 0
        self.set = False

    def __inaccessible_state_vals(self):
        self.access = False
        self.reset = None
        self.actions = None

    def __accessible_state_vals(self):
        self.access = True
        self.reset = False

    def __reset_state_vals(self):
        self.access = True
        self.reset = True
        self.actions = None

    def __get_id_name(self, id_val: str):
        state_names = {
            # Inaccessible States
            'W':'Wall',  # border
            # Accessible States
            'S':'Start',  # map, initialization
            'P':'Path',  # map, neutral
            'F':'Fine',  # map, -reward
            'B':'Bonus', # map, +reward
            'T':'Tunnel', # map, connected
            # Reset States
            'R':'Ravine', # border
            'G':'Goal', # map, termination
            'D':'Ditch', # map, termination
        }
        return state_names[id_val]

    def __state_overwrite(self):
        if not self.set:
            self.set = True
        else:
            state = self.__get_id_name(self.id)
            print(f'{state} State properties at location {self.loc} has been modified.')

    def modify_state(self, id_val: str, location: tuple, reward: tuple, actions: int):
        if id_val is 'W':
            self.__inaccessible_state_vals()
        elif id_val is 'R' or id_val is 'G' or id_val is 'D':
            self.__reset_state_vals()
        else:
            self.__accessible_state_vals()
        # all other consistent values
        self.loc = location
        self.id = id_val
        self.reward = reward
        self.actions = actions
        self.__state_overwrite()

    def get_reward(self) -> float:
        """
        Returns scalar reward to agent based on random.gauss()
        :return: (float) reward from state
        """
        mean = self.reward[0]
        std = math.sqrt(self.reward[1])
        return random.gauss(mean, std)


class GridMap:
    """
    Generates rectangular map (matrix with Grid States) for Grid World with blank default and user-defined rewards,
    obstacles, boundaries or ravines, and goals or trap terminal states, and more.
    """
    # TODO no error control, add/remove states from lists
    path_states = []  # Map states not including termination states (and border)
    map_states = []  # Map states
    start_state = None
    const_reward = 0
    border_reset = 'random'
    termination_reset = 'start'

    def __init__(self, x_size: int = 5, y_size: int = 5):
        """
        Constructor for useless rectangular map of Grid World. Extra features must be added specifically after by user.

        :param x_size: (int > 4) horizontal dimension of Map (relates to # rows of map matrix)
        :param y_size: (int > 4) vertical dimension of Map (relates to # columns of map matrix)
        """
        self.world_shape = (x_size+2, y_size+2)  # +2 accounts for perimeter on all sides
        self.world = self.__init_world()

    # ----------------------------------- GRIDWORLD CREATION AND MODIFICATION -----------------------------------
    def __init_world(self):
        """
        Creates list/matrix filled with default States to represent empty Grid Map.

        :return: (list) Initialized World of default State objects
        """
        x_len, y_len = self.world_shape[0], self.world_shape[1]
        world = []  # initialize empty Map list/matrix
        for r in range(x_len):  # for all rows (x)
            row = []  # initialize temporary row list/matrix
            for c in range(y_len):  # for all column indices (y)
                row.append(GridState())  # build single row of default State objects instances
            world.append(row)  # append full row to Map
        return world

    def _world_update(self, traversal_type: str, id_val: str, reward: tuple, actions: int = 00000000, state_list: list = None):
        """
        Helper function to traverse specific regions of World for defining state value updates

        :param traversal_type: 'map', 'border', 'specific'
        """
        world, x_len, y_len = self.world, self.world_shape[0], self.world_shape[1]

        # Iterate through entire world making updates
        if traversal_type is 'map':
            for r in range(x_len):  # for all rows (x)
                for c in range(y_len):  # for all column indices (y)
                    if not ((r is 0) or (r is x_len-1) or (c is 0) or (c is y_len-1)):
                        # values update to specific state
                        state = self.get_state((r, c))
                        state.modify_state(id_val, (r, c), reward, actions)

        # Iterate only through entire outside perimeter (border) of world making updates
        elif traversal_type is 'border':
            for r in range(x_len):  # for all rows (x)
                for c in range(y_len):  # for all column indices (y)
                    if (r is 0) or (r is x_len-1) or (c is 0) or (c is y_len-1):  # only access border of World
                        # values update to specific state
                        state = self.get_state((r, c))
                        state.modify_state(id_val, (r, c), reward, actions)

        # Iterate only through specific states for making updates
        elif traversal_type is 'specific':
            for state_loc in state_list:
                # values update to specific state
                state = self.get_state(state_loc)
                # all accessible states in map
                state.modify_state(id_val, state_loc, reward, actions)
        else:
            print('WRONG traversal_type entered')

        self.__update_map_lists()

    def __update_map_lists(self):
        """
        Resets and Updates Path List and Map List everytime change is made
        :return:
        """
        # reset lists
        self.path_states.clear()
        self.map_states.clear()
        # update lists
        x_len, y_len = self.world_shape[0], self.world_shape[1]
        for r in range(x_len):  # for all rows (x)
            for c in range(y_len):  # for all column indices (y)
                state = self.get_state((r, c))
                idv = state.id
                if idv is 'P' or idv is 'G' or idv is 'D' or idv is 'B' or idv is 'F' or idv is 'T' or idv is 'S':
                    self.map_states.append(state)
                    if idv is 'P' or idv is 'S':
                        self.path_states.append(state)
                        if idv is 'S':
                            self.start_state = state

    def set_border(self, border_type: str, reward: tuple):
        """
        Establish border on outside of map

        :param border_type: (str) 'W' = Impassible wall, 'R' = Reset Ravine
        :param reward: (float) reward returned
        """
        self._world_update('border', border_type, reward)

    def _get_reset_state(self, reset_type: str = 'start') -> GridState:
        """
        Returns Start State or random Path State to agent depending on reset type

        :return: State to reinitialize agent at
        """
        if reset_type is 'random':
            return random.choice(self.path_states)
        else:
            return self.start_state

    def set_termination_reset(self, reset_type: str):
        self.termination_reset = reset_type

    def set_border_reset(self, reset_type: str):
        self.border_reset = reset_type

    def set_state(self, state_loc_list: list, id_val: str, reward_distr: tuple, actions: int = 00000000):
        """
        Set specific or all states with consistent given values

        :param state_loc_list: (list) list of (r,c) or (x,y) state tuples, or leave list empty [] for ALL STATES
        :param id_val: (str) State ID value
        :param reward_distr: (tuple) (mean, var) reward
        :param actions: (int) 8-bit action encoding
        """
        if len(state_loc_list) is 0:  # ALL States, Map
            self._world_update('map', id_val, reward_distr, actions)
        else:
            self._world_update('specific', id_val, reward_distr, actions, state_loc_list)

    def get_state(self, state_loc: tuple) -> GridState:
        x, y = state_loc[0], state_loc[1]
        return self.world[x][y]

    def set_const_reward(self, reward: int):
        self.const_reward = reward

    def set_reward(self, state, reward_distr: tuple):
        """
        :param reward_distr: (tuple) (mean, variance)
        """
        state.reward = reward_distr

    def print_data(self, data: str = 'id'):
        x_len, y_len = self.world_shape[0], self.world_shape[1]
        print("------------")
        for c in range(y_len):  # for all column indices (y)
            c = y_len-1-c
            for r in range(x_len):  # for all rows (x)
                if data is 'id':
                    print(self.get_state((r, c)).id, end=" ")
                elif data is 'value':
                    print(round(self.get_state((r, c)).value, 2), end=" ")
                elif data is 'reward':
                    print(self.get_state((r, c)).reward, end=" ")
                elif data is 'location':
                    print(self.get_state((r, c)).loc, end=" ")
                elif data is 'reset':
                    print(self.get_state((r,c)).reset, end=" ")
            print("")

    # ----------------------------------- AGENT INTERACTION -----------------------------------

    def get_reward(self, state: GridState):
        return self.const_reward + state.get_reward()

    def initialize_value(self, value: tuple):
        world, x_len, y_len = self.world, self.world_shape[0], self.world_shape[1]
        # Iterate through entire world making updates
        for r in range(x_len):  # for all rows (x)
            for c in range(y_len):  # for all column indices (y)
                if not ((r is 0) or (r is x_len) or (c is 0) or (c is y_len)):
                    # values update to specific state
                    state = self.get_state((r, c))
                    state.value = value

        state.value = value

    def get_value(self, state: GridState):
        return state.value

    def set_value(self, state: GridState, value: float):
        state.value = value

    def initialize_state(self, initialize_type: str = 'start') -> GridState:
        return self._get_reset_state(initialize_type)

    def return_state(self, state: GridState, action: tuple) -> GridState:
        # TODO fix action add action restrictions
        current_loc = state.loc
        x_action, y_action = action[0], action[1]
        next_loc = (current_loc[0] + x_action, current_loc[1] + y_action)
        next_state = self.get_state(next_loc)
        if next_state.id is 'R':  # termination on border (Ravine)
            return self._get_reset_state(self.border_reset)
        elif state.id is 'G' or state.id is 'D':  # termination on path (Goal or Ditch)
            return self._get_reset_state(self.termination_reset)
        elif state.id is 'W':  # stay in place on border or path (Wall)
            return state
        # TODO add tunneling capabilities
        # elif state.id is 'T':  # tunnel pair
        else:
            return next_state # return next path state



