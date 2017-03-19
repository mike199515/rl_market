
class Game(object):
    """
    generic game interface
    """

    def __init__(self):
        pass

    def reset(self):
        """
        reset the game environment
        """
        pass

    def get_observation(self):
        """
        return the observation of current state
        """

    def step(self,action):
        """
        given action, get to the next state, and return (reward, done)
        """
