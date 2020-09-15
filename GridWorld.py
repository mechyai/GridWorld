import numpy as np

class GridWorld:
    """
    Generates rectangular map of Grid World via NumPy matrix with default and user-defined rewards, obstructions,
    boundaries or ravines, and goals or trap terminal states, and more.
    """
    def __init__(self, x_size: int = 5, y_size: int = 5, perimeter: str = 'boundary') -> None:
        """
        Constructor for useless rectangular map of Grid World. Extra features must be added specifically after by user.

        For Reset Layer - a 0 value at index means do not reset at indexed state, a 1 does mean reset
        For Reward Layer - an int value at index refers to reward value at indexed state
        For Path Layer - a 0 value at index means then agent cannot be at  indexed state, a 1 means it can
        For Action Layer - a 4-bit binary value at index refers to {UP,DOWN,LEFT,RIGHT} actions available to agent at
                            indexed state
        :param x_size: (int > 4) horizontal dimension of map (relates to rows of map matrix)
        :param y_size: (int > 4) vertical dimension of map (relates to columns of map matrix)
        :param perimeter: (str) 'boundary' input refers to a perimeter the agent cannot pass onto w/o eps. termination
                                'ravine' input refers to perimeter agent can pass onto w/ eps. termination
        """
        self.map_shape = (x_size+2, y_size+2)  # +2 accounts for perimeter on all sides
        self.x_map = x_size
        self.y_map = y_size
        self.perimeter = perimeter
        self.reset_mtrx = self.__initialize_reset_mtrx()
        self.reward_mtrx = self.__init_mtrx(0)  # 0 -> initialize 0 reward for all states
        self.path_mtrx = self.__init_mtrx(1)  # 1 -> initialize path state for all states
        self.action_mtrx = np.full(self.map_shape, 1111)  # 1111 -> initialize all actions (U,D,L,R) for all states
        self.id_mtrx = np.empty(self.map_shape)

    def __padding(self, mtrx: np.array, pad_val: int):
        """
        Class helper function to create single layer of padding for a given value around a given matrix.

        :param mtrx: (np.ndarray) matrix to have padding added
        :param pad_val: (int) single value used for padding
        """
        mtrx[0, :] = pad_val  # top padding
        mtrx[-1, :] = pad_val  # bottom padding
        mtrx[:, 0] = pad_val  # left padding
        mtrx[:, -1] = pad_val  # right padding

    def __init_mtrx(self, init_val: int = 0) -> np.ndarray:
        """
        Class helper function to create basic map matrix.

        :param init_val: (int 0,1) 0 will create matrix of zeros. 1 will create matrix of ones.
        :return: (np.ndarray) new generic matrix of either zeros or ones
        """
        if init_val is 0:
            mtrx = np.zeros(self.map_shape)
        elif init_val is 1:
            mtrx = np.ones(self.map_shape)
        else:
            mtrx = self.__init_mtrx(0)
            raise ValueError('Argument input must be 1 or 0, program default to 0')
        return mtrx

    def __initialize_reset_mtrx(self) -> np.ndarray:
        """
        Creates initial Reset Layer of Grid World. Includes perimeter (boundary or ravine) and adjusts Path Layer
        accordingly.
        :return: (np.ndarray) initialized Reset Matrix
        """
        rst_mtrx = self.__init_mtrx(0)
        if self.perimeter is 'boundary':
            self.__padding(self.path_mtrx, 0)  # agent cannot pass onto perimeter
        elif self.perimeter is 'ravine':
            self.__padding(rst_mtrx, 1)  # agent does reset with perimeter interaction
        return rst_mtrx







