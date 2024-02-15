"""
Utility for Saving Data to Text File
"""

import datetime
import json
import os
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def save_to_json(
    data_input: pd.DataFrame,
    result: NDArray[np.float64],
) -> None:
    """
    Save input and output data to a json file. A single json file is used
    and new data is appended to the end to log each api call.

    :param pd.DataFrame data_input: The input data to be saved.
    :param NDArray[np.float64] result: The result data to be saved.
    """
    os.makedirs("app/logs", exist_ok=True)
    file_path = "app/logs/model_predictions.json"
    new_data = {
        "input": data_input.to_dict(orient="records"),
        "result": result,
        "date": f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S.%f}",
    }
    try:
        with open(file_path, "r+", encoding="utf-8") as stream:
            # Load current logs
            try:
                data = json.load(stream)
                if not isinstance(data, list):
                    data = [data]
            except json.JSONDecodeError:
                data = []

            # Append new data and write back to file
            data.append(new_data)
            stream.seek(0)
            json.dump(data, stream, indent=4)
            stream.truncate()

    except FileNotFoundError:
        # If the file does not exist, create it and write the new data
        with open(file_path, "w", encoding="utf-8") as stream:
            json.dump([new_data], stream, indent=4)


def read_log_entries(
    limit: int,
) -> list[dict[str, Any]]:
    """
    Read and return a list of log entries from the JSON log file.

    :param int limit: The maximum number of log entries to retrieve.
    :return list[dict[str, Any]]: A list of log entries as dictionaries.
    """
    try:
        loc = "app/logs/model_predictions.json"
        with open(loc, "r", encoding="utf-8") as log_file:
            log_data = json.load(log_file)
        limit = max(1, min(limit, len(log_data)))
        log_data.reverse()
        return log_data[:limit]  # type: ignore
    except FileNotFoundError:
        return []
