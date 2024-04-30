import numpy as np

class StateActionFeatureVectorWithTile():
    def __init__(self,
                 state_low: np.array,
                 state_high: np.array,
                 num_actions: int,
                 num_tilings: int,
                 tile_width: np.array
                 ):
        """
        state_low: possible minimum value for each dimension in state
        state_high: possible maximum value for each dimension in state
        renormalize: renormalizing the second dimension to [-pi / 2, pi / 2] using arctan
        num_actions: the number of possible actions
        num_tilings: # tilings
        tile_width: tile width for each dimension
        """
        self.tile_width, self.num_tilings, self.num_actions = tile_width, num_tilings, num_actions
        self.tiles_dim = np.ceil((state_high - state_low) / tile_width) + 1
        self.start_positions = []
        for i in range(num_tilings):
            self.start_positions.append(state_low - i / num_tilings * tile_width)

    def feature_vector_len(self) -> int:
        """
        return dimension of feature_vector: d = num_actions * num_tilings * num_tiles
        """
        return int(np.product(self.tiles_dim)) * self.num_tilings * self.num_actions

    def __call__(self, s, a) -> np.array:
        """
        implement function x: S+ x A -> [0,1]^d
        if done is True, then return 0^d
        """
        d = self.feature_vector_len()
        answer = np.zeros((d))
        dims = np.concatenate((np.array([self.num_actions, self.num_tilings]), self.tiles_dim))
        for i in range(self.num_tilings):
            current_position = np.floor((s - self.start_positions[i]) / self.tile_width)
            current_position = np.concatenate((np.array([a, i]), current_position))
            index = np.ravel_multi_index(current_position.astype(int), dims.astype(int))
            answer[index] = 1
        return answer