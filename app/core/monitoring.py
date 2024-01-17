import pandas as pd
import numpy as np
import json
import datetime


def save_to_text(input: pd.DataFrame, result: np.array) -> None:
    """
    Saves input/output dicts to bigquery
    """
    file_path = "app/log/model_predictions.json"
    new_data = {
        "input": input.to_dict(orient="records"),
        "result": result[0],
        "date": f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S.%f}",
    }
    try:
        with open(file_path, "r+") as stream:
            # Load existing data
            try:
                data = json.load(stream)
                if not isinstance(data, list):
                    # If data is not a list, make it a list
                    data = [data]
            except json.JSONDecodeError:
                # If the file is empty and causing JSONDecodeError, start a new list
                data = []

            # Append new data and write back to file
            data.append(new_data)
            stream.seek(0)
            json.dump(data, stream, indent=4)
            stream.truncate()

    except FileNotFoundError:
        # If the file does not exist, create it and write the new data
        with open(file_path, "w") as stream:
            json.dump([new_data], stream, indent=4)
