from abc import ABC

import pandas as pd


def multiton(cls):
    """
    Multiton decorator for any given class
    """
    instances = {}

    def getinstance(*args, **kwargs):
        key = (
            frozenset(make_hashable(arg) for arg in args),
            frozenset(make_hashable(kwarg) for kwarg in kwargs)
        )
        print(key)
        if key not in instances:
            instances[key] = cls(*args, **kwargs)
        return instances[key]
    return getinstance


def make_hashable(obj):
    """
    Make an object hashable.
    Supported types: dict, list, pandas DataFrame
    """
    if isinstance(obj, dict):
        return frozenset((key, make_hashable(value)) for key, value in obj.items())
    elif isinstance(obj, list):
        return tuple(make_hashable(item) for item in obj)
    elif isinstance(obj, pd.DataFrame):
        return obj.to_string()
    else:
        return obj


class BasePredictor(ABC):
    def __init__(self, data: pd.DataFrame, config: dict):
        self.data = data
        self._data_components = config['data_components'])
        self._predictor_components = config['predictor_components']


class CustomPredictor(BasePredictor):
    """
    CustomPredictor class with a multiton pattern (alternative for multiton decorator).
    A CustomPredictor instance will hold multiple components.
    These features will be instantiated from factory classes, which will be called
    by a PredictorBuilder instance.
    """

    _instances = {}

    # Multiton pattern
    def __new__(cls, data: pd.DataFrame, config: dict):
        key = (make_hashable(data), make_hashable(config))

        if key not in cls._instances:
            instance = super(CustomPredictor, cls).__new__(cls)
            instance._initialized = False
            cls._instances[key] = instance

        return cls._instances[key]

    def __init__(self, data: pd.DataFrame, config: dict):
        if self._initialized:
            return

        self._initialized = True
        super().__init__(data, config)
