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
        path = Path(os.getenv("ML_MODELS_DIR"), "trained_model", "model.pkl")
        if not path.exists():
            message = f"Machine learning model at {path} doesn't exist"
            logger.warning(message)
            raise FileNotFoundError(message)

        try:
            with path.open("rb") as file:
                model = load_wrapper(file)
        except Exception as e:
            message = f"Model could not be loaded due to error: {e}"
            logger.error(message)
            raise ModelLoadException(message) from e

        if model is None:
            message = "Failed to load the model, it is None."
            logger.error(message)
            raise ModelLoadException(message)

        logger.info(f"Succesfully loaded model: {path.resolve()}")
        return model
