from typing import Literal

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from src.infrastructure.config.settings import get_settings
from src.interface.http.middleware.auth import create_token

router = APIRouter(prefix="/auth", tags=["auth"])


class TokenRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=100)
    role: Literal["user", "admin"] = "user"
    admin_secret: str = Field(default="")


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str


@router.post("/token", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def issue_token(body: TokenRequest) -> TokenResponse:
    """Issue a signed JWT. Admin tokens require a valid ADMIN_SECRET in the request body."""
    settings = get_settings()
    if body.role == "admin":
        # pylint: disable=no-member
        if not settings.admin.secret or body.admin_secret != settings.admin.secret:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Invalid admin secret")
    token = create_token(body.user_id, body.role, settings.jwt)
    return TokenResponse(access_token=token, role=body.role)
