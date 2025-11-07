import os
import re
from typing import Any
from pathlib import Path
import yaml
from scipy.io import loadmat
import numpy as np
import pandas as pd

# YAML loader that is able to read scientific notation
custom_loader = yaml.SafeLoader
custom_loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        """^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*?
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$""",
        re.X,
    ),
    list("-+0123456789."),
)

def load_yaml(path: str | Path, fname: str | Path) -> Any:
    """Loads and returns the contents of the YAML file.

    Parameters
    ----------
    path : str | Path
        Path to the file to be loaded.
    fname : str | Path
        Name of the file (ending in .yaml) to be loaded.

    Returns
    -------
    Any
        Whatever content is in the YAML file.
    """
    return yaml.load(open(os.path.join(path, fname)), Loader=custom_loader)


def load_surrogate_mat(filepath):
    """
    Loads a surrogate model from a .mat file and converts it into a nested dictionary format,
    preserving MATLAB struct-like access in Python.

    Parameters
    ----------
    filepath : str
        Path to the .mat file containing the surrogate model.

    Returns
    -------
    dict
        Nested dictionary containing surrogate model data, accessible similarly to MATLAB structs.
    """
    # Load the .mat file
    mat_data = loadmat(filepath, struct_as_record=False, squeeze_me=True)

    # Filter out MATLAB metadata entries (those starting with '__')
    surrogate_data = {key: value for key, value in mat_data.items() if not key.startswith('__')}

    # Recursive function to convert MATLAB structs, cell arrays, arrays, and matrices into nested dictionaries and lists
    def mat_to_dict(mat_obj):
        # Check if the object is a MATLAB struct (np.void or mat_struct)
        if isinstance(mat_obj, np.void):
            # Convert struct fields to dictionary entries
            return {field: mat_to_dict(mat_obj[field]) for field in mat_obj.dtype.names}
        
        # Check if the object is a `mat_struct` (used in older versions of scipy)
        elif hasattr(mat_obj, '_fieldnames'):  
            return {field: mat_to_dict(getattr(mat_obj, field)) for field in mat_obj._fieldnames}
        
        # Check if it's a cell array or an array of structs (numpy.ndarray)
        elif isinstance(mat_obj, np.ndarray):
            if mat_obj.dtype.names:  # Structured array with named fields (array of structs)
                return [mat_to_dict(mat_obj[i]) for i in range(mat_obj.size)]
            else:  # General array or cell array
                if mat_obj.ndim >= 3:  # For 3D+ arrays, keep as NumPy array for efficiency
                    return mat_obj
                elif mat_obj.ndim == 2:  # For 2D arrays (e.g., 3x3 matrix)
                    return mat_obj.tolist()  # Convert to nested lists for easy access
                else:
                    return [mat_to_dict(item) for item in mat_obj] if mat_obj.ndim == 1 else mat_obj.tolist()
        
        # Directly return scalars, strings, or other non-structured elements
        return mat_obj

    # Convert all top-level data structures to nested dictionaries
    for key in surrogate_data:
        surrogate_data[key] = mat_to_dict(surrogate_data[key])

    print(f"Loaded surrogate model data from {filepath} as nested dictionaries with NumPy array representation for 3D+ matrices.")
    return surrogate_data



def process_duration_fields(data):
        """
        Recursively processes dictionaries and lists to find 'value'/'unit' pairs 
        and convert them into hours.

        Parameters
        ----------
        data : dict or list
            The input data structure to process.

        Returns
        -------
        dict or list
            The processed data structure with durations converted to hours.
        """
        if isinstance(data, dict):
            new_data = {}
            for key, value in data.items():
                if isinstance(value, dict) and 'value' in value and 'unit' in value:
                    try:
                        duration_hours = calculate_duration_in_hours(value['value'], value['unit'])
                        new_data[key] = value  # Preserve the original structure
                        new_data[f"{key}_h"] = duration_hours
                    except ValueError as e:
                        print(f"Error processing {key}: {e}")
                else:
                    new_data[key] = process_duration_fields(value)  # Recurse into sub-dictionaries/lists
            return new_data
        elif isinstance(data, list):
            return [process_duration_fields(item) for item in data]
        else:
            return data


def calculate_duration_in_hours(duration_value: int, duration_unit: str) -> int:
    """Convert project duration to hours based on the provided unit."""
    duration_unit = duration_unit.lower()
    if duration_unit == "years":
        return duration_value * 365 * 24
    elif duration_unit == "months":
        return duration_value * 30 * 24
    elif duration_unit == "days":
        return duration_value * 24
    elif duration_unit == "hours":
        return duration_value
    else:
        raise ValueError(f"Unsupported duration unit: {duration_unit}")
    

def loadcsv(folder_path, filename, **kwargs):
    """
    Loads a CSV file and returns a pandas DataFrame.
    
    Parameters:
        folder_path (str): The path to the folder containing the CSV file.
        filename (str): The name of the CSV file.
        **kwargs: Additional arguments passed to pandas.read_csv (e.g., delimiter, encoding).
    
    Returns:
        pd.DataFrame: The loaded DataFrame.
    
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be loaded as a DataFrame.
    """
    full_path = os.path.join(folder_path, filename)
    
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"CSV file not found: {full_path}")
    
    try:
        df = pd.read_csv(full_path, **kwargs)
    except Exception as e:
        raise ValueError(f"Error loading CSV file '{filename}': {e}")
    
    return df

