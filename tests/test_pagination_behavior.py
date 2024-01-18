"""
In order to test behavior of pagination function
"""
import pytest

from app.core.paginator import pagination, InvalidPageNumberError


def test_pagination_400_initial_default() -> None:
    """
    Test pagination function behavior with 400 total items and default
    initial page 1.
    """
    d = pagination(1, 20, 400, list(range(400)))
    print(d["listings"])
    print(list(range(20)))
    assert d["listings"] == list(range(0, 20))


def test_pagination_400_10th_page() -> None:
    """
    Test pagination function behavior with 400 total items on the 10th page.
    """
    d = pagination(10, 20, 400, list(range(400)))
    print(d["listings"])
    print(list(range(180, 200)))
    assert d["listings"] == list(range(180, 200))


def test_pagination_400_start_0() -> None:
    """
    Test pagination function behavior with 400 total items, starting from page 0.
    """

    d = pagination(19, 20, 400, list(range(400)), start_page_as_1=False)
    print(d["listings"])
    print(list(range(380, 400)))
    assert d["listings"] == list(range(380, 400))


def test_pagination_400_start_1() -> None:
    """
    Test pagination function behavior with 400 total items and starting from page 1.
    """
    d = pagination(20, 20, 400, list(range(400)))
    print(d["listings"])
    print(list(range(380, 400)))
    assert d["listings"] == list(range(380, 400))


def test_pagination_400_set_start_1_equals_true_and_init_as_pagenumber_as_0() -> None:
    """
    Test pagination function behavior with 400 total items, starting from
    page 0, and 'start_page_as_1' set to True.
    """
    with pytest.raises(InvalidPageNumberError, match=r".* start > 0. *"):
        _ = pagination(0, 20, 400, list(range(400)))
