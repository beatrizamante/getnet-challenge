import time
from typing import Literal

import jwt
from fastapi import HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from src.infrastructure.config.settings import JWTSettings, get_settings

_bearer = HTTPBearer(auto_error=True)


class TokenClaims(BaseModel):
    sub: str
    role: Literal["user", "admin"]


def create_token(user_id: str, role: str, settings: JWTSettings) -> str:
    payload = {
        "sub": user_id,
        "role": role,
        "exp": int(time.time()) + settings.expiry_minutes * 60,
    }
    return jwt.encode(payload, settings.secret, algorithm=settings.algorithm)


def _verify(token: str) -> TokenClaims:
    settings = get_settings().jwt
    try:
        # pylint: disable=no-member
        payload = jwt.decode(token, settings.secret, algorithms=[settings.algorithm])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    return TokenClaims(sub=payload["sub"], role=payload["role"])


class _RequireRole:
    """FastAPI dependency: validates Bearer JWT and enforces role membership."""

    def __init__(self, *roles: str) -> None:
        self._roles = frozenset(roles)

    def __call__(
        self,
        creds: HTTPAuthorizationCredentials = Security(_bearer),
    ) -> TokenClaims:
        claims = _verify(creds.credentials)
        if claims.role not in self._roles:
            raise HTTPException(status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
        return claims


# Singletons shared by routes and test overrides — identity matters for dependency_overrides
require_user = _RequireRole("user", "admin")
require_admin = _RequireRole("admin")
