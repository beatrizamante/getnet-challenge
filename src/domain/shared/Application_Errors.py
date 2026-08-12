class BaseError(Exception):
    """Base error for all application-level exceptions."""

    code = "BASE_ERROR"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class UnauthorizedError(BaseError):
    code = "UNAUTHORIZED_ERROR"


class NotFoundError(BaseError):
    code = "NOT_FOUND_ERROR"


class UserNotFoundError(NotFoundError):
    code = "USER_NOT_FOUND_ERROR"

    def __init__(self, user_id: str) -> None:
        super().__init__(f"User '{user_id}' not found.")
        self.user_id = user_id
