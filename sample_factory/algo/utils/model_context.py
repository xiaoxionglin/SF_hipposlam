from typing import Dict

from sample_factory.algo.learning.learner_factory import LearnerFactory
from sample_factory.model.model_factory import ModelFactory


class SampleFactoryModelContext:
    def __init__(self):
        self.model_factory = ModelFactory()
        self.learner_factory = LearnerFactory()


GLOBAL_MODEL_CONTEXT = None


def sf_global_model_context() -> SampleFactoryModelContext:
    global GLOBAL_MODEL_CONTEXT
    if GLOBAL_MODEL_CONTEXT is None:
        GLOBAL_MODEL_CONTEXT = SampleFactoryModelContext()
    return GLOBAL_MODEL_CONTEXT


def set_global_model_context(ctx: SampleFactoryModelContext):
    global GLOBAL_MODEL_CONTEXT
    GLOBAL_MODEL_CONTEXT = ctx


def reset_global_model_context():
    """
    Most useful in tests, call this after any part of the global context has been modified
    by a test in any way.
    """
    global GLOBAL_MODEL_CONTEXT
    GLOBAL_MODEL_CONTEXT = SampleFactoryModelContext()


def global_model_factory() -> ModelFactory:
    """
    :return: global model factory
    :rtype: ModelFactory
    """
    return sf_global_model_context().model_factory


def global_learner_factory() -> LearnerFactory:
    """
    :return: global learner factory
    :rtype: LearnerFactory
    """
    return sf_global_model_context().learner_factory
