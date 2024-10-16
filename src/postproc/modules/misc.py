from __future__ import annotations

from numba import jit, types
from numba.typed import List


def infer_numba_type_and_depth(py_elem, depth=0):
    """
    Recursively infer the Numba type of the given Python element.
    Handles nested lists by creating corresponding Numba List types.
    """
    if isinstance(py_elem, int):
        return types.int64, depth
    if isinstance(py_elem, float):
        return types.float64, depth
    if isinstance(py_elem, list):
        if not py_elem:  # Handle empty list
            return None, depth
        for elem in py_elem:
            elem_type, new_depth = infer_numba_type_and_depth(elem, depth + 1)
            if elem_type is not None:
                return types.ListType(elem_type), new_depth  # Return the list type here
        return None, depth
    text = f"Unsupported type: {type(py_elem)}"
    raise TypeError(text)


# @njit
@jit(forceobj=True, looplift=False)
def _convert_to_numba_list(py_list, elem_type, depth):
    """
    Recursively convert a Python list to a Numba typed List.
    This is a helper function for numba-compiled conversion.
    """
    # Create an empty Numba List of the given element type
    numba_list = List.empty_list(elem_type.dtype)

    # Determine the type of elements and populate the Numba list
    if depth == 1:  # Base case for the innermost list
        for elem in py_list:
            if isinstance(elem, int):
                numba_list.append(int(elem))  # Append int to numba_list
            elif isinstance(elem, float):
                numba_list.append(float(elem))  # Append float to numba_list
            else:
                text = f"Unsupported type in innermost list: {type(elem)}"
                raise TypeError(text)
    else:  # Handle nested lists
        for elem in py_list:
            if isinstance(elem, list):
                # Recursively convert the nested list
                nested_list = _convert_to_numba_list(elem, elem_type.dtype, depth - 1)
                numba_list.append(nested_list)  # Append the nested list
            else:
                text = f"Expected a list for depth {depth}, but got: {type(elem)}"
                raise TypeError(text)

    return numba_list


def python_list_to_numba_list(py_list):
    """
    Convert a Python list to a Numba typed List, preserving nested structures.
    Calls an Numba-compiled function for efficiency.
    """
    if len(py_list) == 0:
        text = "Cannot convert an empty list"
        raise ValueError(text)

    # Infer the Numba type for the elements
    elem_type, depth = infer_numba_type_and_depth(py_list)
    # Use Numba-compiled function for conversion
    return _convert_to_numba_list(py_list, elem_type, depth)
