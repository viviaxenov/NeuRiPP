import abc

from ..parametric_pushforward.parametric_pushforward import ParametricPushforward


class FixedPointProblem:
    def __init__(self, model: ParametricPushforward, *args, **kwargs):
        self._model = model
        pass
    
    @abc.abstractmethod
    def residual(self, params):
    """ Gives a current fixed-point residual in the problem"""
        pass

    @abc.abstractmethod
    def suggest_initial(self, *args, **kwargs):
        pass

    @abc.abstractproperty
    def metrics(self):
        """A list of (Callable, str) that evaluate the current solution""" 
        return []

