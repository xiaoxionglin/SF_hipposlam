from typing import Callable

from torch import Tensor

from sample_factory.algo.learning.learner import BaseLearner, default_make_learner_func
from sample_factory.algo.utils.env_info import EnvInfo
from sample_factory.algo.utils.model_sharing import ParameterServer
from sample_factory.utils.typing import Config, PolicyID
from sample_factory.utils.utils import log

MakeLearnerFunc = Callable[[Config, EnvInfo, Tensor, PolicyID, ParameterServer], BaseLearner]


class LearnerFactory:
    def __init__(self):
        """
        Optional custom functions for creating parts of the model (encoders, decoders, etc.), or
        even overriding the entire actor-critic with a custom model.
        """

        self.make_learner_func: MakeLearnerFunc = default_make_learner_func

    def register_learner_factory(self, make_learner_func: MakeLearnerFunc):
        """
        Override the default learner with a custom model.
        """
        log.debug(f"register_learner_factory: {make_learner_func}")
        self.make_learner_func = make_learner_func
