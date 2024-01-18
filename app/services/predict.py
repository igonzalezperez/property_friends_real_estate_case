"""
This module defines the ModelHandlerScore class for handling machine learning
models.
"""
import os
import typing
from pathlib import Path

import pandas as pd
from loguru import logger

# pylint: disable=E0401
from app.core.config import ML_MODELS_DIR, MODEL_NAME
from app.core.errors import ModelLoadException, PredictException


class ModelHandlerScore:
    """
    Class for handling machine learning models and making predictions.
    """

    model = None

    @typing.no_type_check
    @classmethod
    def predict(
        cls,
        data_input: pd.DataFrame,
        load_wrapper=None,
        method: str = "predict",
    ):
        """
        Predicts using the loaded model.

        :param pd.DataFrame data_input: Input data for prediction.
        :param load_wrapper: A function to load the model. Defaults to None
        :param str method: The prediction method to use. Defaults to predict
        :raises PredictException: If the prediction method is missing.
        :return: The prediction result.
        """
        clf = cls.get_model(load_wrapper)
        if hasattr(clf, method):
            return getattr(clf, method)(data_input)
        raise PredictException(f"'{method}' attribute is missing")

    @typing.no_type_check
    @classmethod
    def get_model(cls, load_wrapper):
        """
        Gets the loaded model.

        :param load_wrapper: A function to load the model.
        :return: The loaded model.
        """
        if cls.model is None and load_wrapper:
            cls.model = cls.load(load_wrapper)
        return cls.model

    @typing.no_type_check
    @staticmethod
    def load(load_wrapper):
        """
        Loads the machine learning model.

        :param load_wrapper: A function to load the model.
        :raises FileNotFoundError: If the model file is not found.
        :raises ModelLoadException: If the model fails to load.
        :return: The loaded model.
        """
        model = None
        path = Path(ML_MODELS_DIR, MODEL_NAME)
        if not os.path.exists(path):
            message = f"Machine learning model at {path} not exists!"
            logger.error(message)
            raise FileNotFoundError(message)
        model = load_wrapper(path)
        if not model:
            message = f"Model {model} could not load!"
            logger.error(message)
            raise ModelLoadException(message)

        logger.info(f"Succesfully loaded model: {Path(path).resolve()}")
        return model
