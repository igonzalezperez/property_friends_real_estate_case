"""
Custom FastAPI authentication scheme for API token validation.
"""
from fastapi import HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.middleware.api_key_auth import ApiKeyManager


# pylint: disable=too-few-public-methods
class TokenValidator(HTTPBearer):
    """
    HTTPBearer inherited class for API token validation.

    :param bool auto_error: If True, raises an HTTPException on token
    validation failure. Defaults to True
    :raises HTTPException: Raised when token validation fails
    :return HTTPAuthorizationCredentials: Valid API token credentials
    """

    def __init__(self, auto_error: bool = True):
        super().__init__(auto_error=auto_error)

    async def __call__(self, request: Request) -> HTTPAuthorizationCredentials:
        """
        Validate API token using the ApiKeyManager.

        :param Request request: FastAPI request object
        :raises HTTPException: Raised if token validation fails
        :return HTTPAuthorizationCredentials: Valid API token credentials
        """
        credentials = await super().__call__(request)
        if credentials:
            token = credentials.credentials
            if ApiKeyManager().validate_api_key(token):
                return credentials
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing API Token",
        )


token_auth_scheme = TokenValidator()
