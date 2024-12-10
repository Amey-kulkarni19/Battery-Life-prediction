import numpy as np
from scipy.io import loadmat

def extract_discharge_data(file_path):
    # Load the .mat file data
    data = loadmat(file_path)
    
    # Get the key of the last item in the data dictionary (it will usually contain the battery data)
    battery_key = list(data.keys())[-1]  

    # Extract all cycles from the loaded data
    cycles = data[battery_key]['cycle'][0, 0][0]  

    # Filter the cycles to only include 'discharge' cycles
    discharge_cycles = [cycle for cycle in cycles if cycle['type'][0] == 'discharge']
    
    # If no discharge cycles are found, raise an error
    if not discharge_cycles:
        raise ValueError("No discharge cycles found.")
    return discharge_cycles

def extract_capacities(discharge_cycles):

     # Extract the capacity data for each discharge cycle
    capacities = [
        float(cycle['data']['Capacity'][0, 0][0]) for cycle in discharge_cycles
    ]

    # Return the capacities as a numpy array
    return np.array(capacities)
