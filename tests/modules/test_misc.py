from __future__ import annotations

import numba
import pytest
from numba import types
from numba.typed import List

from postproc.modules.misc import infer_numba_type_and_depth, python_list_to_numba_list


def test_infer_numba_type_and_depth():
    # Test with integer
    assert infer_numba_type_and_depth(1) == (types.int64, 0)

    # Test with float
    assert infer_numba_type_and_depth(1.0) == (types.float64, 0)

    # Test with nested list
    assert infer_numba_type_and_depth([1, 2, 3]) == (types.ListType(types.int64), 1)
    assert infer_numba_type_and_depth([1.0, 2.0, 3.0]) == (
        types.ListType(types.float64),
        1,
    )
    assert infer_numba_type_and_depth([[1, 2], [3, 4]]) == (
        types.ListType(types.ListType(types.int64)),
        2,
    )
    assert infer_numba_type_and_depth([[[1], [2]], [[3], [4]]]) == (
        types.ListType(types.ListType(types.ListType(types.int64))),
        3,
    )

    # Test with empty list
    with pytest.raises(ValueError, match="Cannot infer type from an empty list"):
        infer_numba_type_and_depth([])

    # Test with unsupported type
    with pytest.raises(TypeError, match="Unsupported type: <class 'str'>"):
        infer_numba_type_and_depth("string")


def test_python_list_to_numba_list():
    # Test with integer list
    py_list = [1, 2, 3]
    numba_list = python_list_to_numba_list(py_list)
    assert isinstance(numba_list, List)
    assert numba_list == List([1, 2, 3])

    # Test with float list
    py_list = [1.0, 2.0, 3.0]
    numba_list = python_list_to_numba_list(py_list)
    assert isinstance(numba_list, List)
    assert numba_list == List([1.0, 2.0, 3.0])

    # Test with nested list
    py_list = [[1, 2], [3, 4]]
    numba_list = python_list_to_numba_list(py_list)
    assert isinstance(numba_list, List)
    assert numba_list == List([List([1, 2]), List([3, 4])])

    # Test with mixed type list (should raise TypeError)
    py_list = [1, 2.0, 3]
    with pytest.raises(numba.core.errors.TypingError):
        python_list_to_numba_list(py_list)

    # Test with empty list
    with pytest.raises(ValueError, match="Cannot convert an empty list"):
        python_list_to_numba_list([])


if __name__ == "__main__":
    pytest.main()
