from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials


class TokenValidator(HTTPBearer):
    def __init__(self, auto_error: bool = True):
        super().__init__(auto_error=auto_error)

    async def __call__(
        self,
        request: Request,
        credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
    ):
        if credentials:
            token = credentials.credentials
            if (
                token
                == "HSHyq6cKqPppj5rcHgszjRImJK4CdvnJ2ODNkGHbDAD-iewP07C2RuuHEaP1oQanqy4"
            ):
                return credentials.credentials
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Invalid or missing API Token"
        )


token_auth_scheme = TokenValidator()
