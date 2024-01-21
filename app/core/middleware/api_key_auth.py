"""
Utilities to handle api key generation, storing and validation
"""
import datetime
import json
import os
import secrets
from pathlib import Path
from typing import Any, Optional

# Create folder to store valid api keys
os.makedirs(Path("app", "valid_keys"), exist_ok=True)


class ApiKeyManager:
    """
    Class to manage api keys
    """

    def __init__(
        self,
        file_path: Path = Path(
            "app",
            "valid_keys",
            "api_keys.json",
        ),
    ):
        """
        Initializes an instance of the ApiKeyManager.

        :param Path file_path: The file path to store API keys. Defaults to
        "app/valid_keys/api_keys.json".
        """
        self.file_path = file_path
        self.api_keys = self.load_api_keys()

    def generate_api_key(self) -> str:
        """
        Generate a new API key.

        :return str: A newly generated API key.
        """
        new_key = secrets.token_urlsafe(32)
        return new_key

    def add_api_key(self, user_id: str) -> str:
        """
        Generate and store a new API key for a given user ID.
        API keys are saved with created at timestamp for rotation:

        Example:
        {"user_id": {"api_key": "api_key_value", "created_at": TIMESTAMP}}

        :param str user_id: The user ID for which to generate the API key.
        :return str: The newly generated API key.
        """
        new_key = self.generate_api_key()
        new_entry = {
            "api_key": new_key,
            "created_at": str(datetime.datetime.now()),
        }
        self.api_keys[user_id] = new_entry
        self.save_api_keys()
        return new_key

    def validate_api_key(self, key: str) -> Optional[str]:
        """
        Validate an API key and return the associated user ID if valid.

        :param str key: The API key to validate.
        :return Optional[str]: The associated user ID if the key is valid, or
        None if not valid.
        """
        for user_id, api_key_info in self.api_keys.items():
            if api_key_info["api_key"] == key:
                return str(user_id)
        return None

    def load_api_keys(self) -> Any:
        """
        Load API keys from a file.

        :return dict: A dictionary of loaded API keys.
        """
        if self.file_path.exists():
            with open(self.file_path, "r", encoding="utf-8") as file:
                return json.load(file)
        return {}

    def save_api_keys(self) -> None:
        """
        Save the current API keys to a file.
        """
        with open(self.file_path, "w", encoding="utf-8") as file:
            json.dump(self.api_keys, file)

    def rotate_api_key(self, _: str) -> None:
        """
        Rotate API key
        """
        # To implement later, placeholder variable "_" for user_id
