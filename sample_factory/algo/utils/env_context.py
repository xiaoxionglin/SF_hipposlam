from typing import Dict

from sample_factory.utils.typing import CreateEnvFunc


class SampleFactoryEnvContext:
    def __init__(self):
        self.env_registry = dict()


GLOBAL_ENV_CONTEXT = None


def sf_global_env_context() -> SampleFactoryEnvContext:
    global GLOBAL_ENV_CONTEXT
    if GLOBAL_ENV_CONTEXT is None:
        GLOBAL_ENV_CONTEXT = SampleFactoryEnvContext()
    return GLOBAL_ENV_CONTEXT


def set_global_env_context(ctx: SampleFactoryEnvContext):
    global GLOBAL_ENV_CONTEXT
    GLOBAL_ENV_CONTEXT = ctx


def reset_global_env_context():
    """
    Most useful in tests, call this after any part of the global context has been modified
    by a test in any way.
    """
    global GLOBAL_ENV_CONTEXT
    GLOBAL_ENV_CONTEXT = SampleFactoryEnvContext()


def global_env_registry() -> Dict[str, CreateEnvFunc]:
    """
    :return: global env registry
    :rtype: EnvRegistry
    """
    return sf_global_env_context().env_registry

