import os
import pandas as pd
import numpy as np
from File_Handling import load_yaml, load_surrogate_mat
from scipy.interpolate import RegularGridInterpolator

class TurbineSurrogateQuery:
    def __init__(self, wind_farm_data):
        self.wind_farm_data = wind_farm_data
        self.surrogate_data = self.load_surrogate_model()

    def load_surrogate_model(self):
        """
        Loads a surrogate model for the wind farm simulation based on file type.
        """
        wf_data = self.wind_farm_data.get('WF', {})
        wf_input_folder = wf_data.get('WF_inputFolder', "")
        wf_input_files = wf_data.get('WF_inputFiles', {})

        surrogate_file = wf_input_files.get('SM', None)

        if surrogate_file and surrogate_file.endswith('.mat'):
            filepath = os.path.join(wf_input_folder, surrogate_file)
            self.surrogate_data = load_surrogate_mat(filepath)
            return self.surrogate_data

        self.surrogate_data = {}
        return self.surrogate_data

    def get_points_from_surrogate_spline(self, Vreq, TIreq, Preq, sensor_names, interp_type, names=None):
        """
        Queries surrogate data using spline interpolation for specified sensors.
        """
        output = {}


        # Use specified names or all available model names if none specified
        model_names = names if names else self.surrogate_data.keys()

        # Iterate over each surrogate model specified
        for model_name in model_names:
            data = self.surrogate_data.get(model_name, None)
            if data is None:
                print(f"Warning: {model_name} not found in surrogate data")
                continue

            # Extract grid dimensions from the nested structure
            try:
                dimensions = data["Dimensions"]
                dim1 = np.array(dimensions["Dim1"][1]).squeeze()  
                dim2 = np.array(dimensions["Dim2"][1]).squeeze()  
                dim3 = np.array(dimensions["Dim3"][1]).squeeze()  
                grid = (dim1, dim2, dim3)

                # Check if grid dimensions are valid
                if any(dim is None for dim in grid):
                    print(f"Error: Invalid grid dimensions in {model_name}")
                    continue

                # Iterate over each field in data and subfields for interpolation
                for key, field_data in data.items():
                    if key not in sensor_names:
                        # Skip keys not in the specified WF_ResponseSensors
                        continue
                    
                    if isinstance(field_data, dict):
                        for subkey, values in field_data.items():
                            if values is None:
                                print(f"Warning: Missing data for {model_name} - {key}_{subkey}")
                                continue

                            try:
                                interpolator = RegularGridInterpolator(grid, values, method=interp_type)
                                query_points = np.array([Vreq, TIreq, Preq]).T
                                interpolated_values = interpolator(query_points)
                                output[f"{key}_{subkey}"] = interpolated_values
                            except Exception as e:
                                print(f"Interpolation error in {model_name} - {key}_{subkey}: {e}")
            except KeyError as e:
                print(f"Error accessing grid dimensions in {model_name}: {e}")
            except Exception as e:
                print(f"Error processing grid dimensions for model {model_name}: {e}")

        return output
