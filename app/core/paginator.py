"""
Pagination Utility

Provides functions for paginating data and generating pagination metadata.
"""
from typing import Any, Dict, Optional


class InvalidPageNumberError(Exception):
    """
    An exception raised when the page number is invalid. Page number must start
    greater than 0 when `start_page_as_1` is True and the page number is
    defined as less than or equal to 0.
    """

    def __init__(self) -> None:
        super().__init__(
            """Page number must start > 0.
            Cause: start_page_as_1=True and page_number defined as <= 0"""
        )


def pagination(
    page_number: int = 1,
    page_size: int = 20,
    total_count: int = 0,
    data: Optional[Any] = None,
    start_page_as_1: bool = True,
) -> Dict[str, Any]:
    """
    Return payload that contains metainformations about
    pagination and listing data.
    page_number starts with 0 (array like),
    if start_page_as_1 defined as True, start with 1.
    """
    if data is None:
        data = []
    if start_page_as_1:
        if page_number <= 0:
            raise InvalidPageNumberError
        page_number -= 1
    remaining = total_count % page_size
    total_pages = (
        total_count // page_size + 1 if remaining else total_count // page_size
    )
    begin = page_number * page_size
    end = begin
    if page_number == total_pages and remaining:
        end += remaining
    else:
        end += page_size
    return {
        "begin": begin,
        "end": end,
        "totalPages": total_pages,
        "remaining": remaining,
        "pageNumber": page_number,
        "pageSize": page_size,
        "totalCount": total_count,
        "listings": data[begin:end],
    }
