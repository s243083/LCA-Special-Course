from typing import TYPE_CHECKING, Any, Callable
from pathlib import Path


import attr
import attrs
import numpy as np
import pandas as pd
from attrs import Factory, Attribute, field, define




@define
class FromDictMixin:
    """A Mixin class to allow for kwargs overloading when a data class doesn't
    have a specific parameter definied. This allows passing of larger dictionaries
    to a data class without throwing an error.

    Raises
    ------
    AttributeError
        Raised if the required class inputs are not provided.
    """

    @classmethod
    def from_dict(cls, data: dict):
        """Map a data dictionary to an `attrs`-defined class.

        TODO: Add an error to ensure that either none or all the parameters are passed

        Parameters
        ----------
            data : dict
                The data dictionary to be mapped.

        Returns
        -------
            cls : Any
                The `attrs`-defined class.
        """
        if TYPE_CHECKING:
            assert hasattr(cls, "__attrs_attrs__")
        # Get all parameters from the input dictionary that map to the class init
        kwargs = {
            a.name: data[a.name]
            for a in cls.__attrs_attrs__
            if a.name in data and a.init
        }

        # Map the inputs that must be provided:
        # 1) must be initialized
        # 2) no default value defined
        required_inputs = [
            a.name
            for a in cls.__attrs_attrs__
            if a.init and isinstance(a.default, attr._make._Nothing)  # type: ignore
        ]
        undefined = sorted(set(required_inputs) - set(kwargs))
        if undefined:
            raise AttributeError(
                f"The class defintion for {cls.__name__} is missing the following"
                f" inputs: {undefined}"
            )
        return cls(**kwargs)
