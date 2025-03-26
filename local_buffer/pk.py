#!/usr/bin/env python3
"""
read_pickle.py

A script to read and print the contents of a pickle file.

Usage:
    python read_pickle.py --file_path path/to/your/file.pickle

Example:
    python read_pickle.py --file_path trajectories/actor_1/traj_0.pickle
"""

import argparse
import pickle
import os
import sys
import numpy as np

def read_pickle_file(file_path):
    """
    Reads a pickle file and returns the loaded object.

    Args:
        file_path (str): Path to the pickle file.

    Returns:
        Any: The Python object stored in the pickle file.
    """
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        sys.exit(1)

    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            return data
    except pickle.UnpicklingError:
        print(f"Error: Failed to unpickle the file '{file_path}'. It may be corrupted or not a pickle file.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while reading '{file_path}': {e}")
        sys.exit(1)

def print_data(data, indent=0):
    """
    Recursively prints the data in a readable format.

    Args:
        data (Any): The data to print.
        indent (int): Current indentation level (used for recursion).
    """

    
    spacing = '  ' * indent
    if isinstance(data, dict):
        print(f"{spacing}{{")
        for key, value in data.items():
            print(f"{spacing}  {repr(key)}: ", end='')
            print_data(value, indent + 2)
        print(f"{spacing}}}")
    elif isinstance(data, list):
        print(f"{spacing}[")
        for item in data:
            print_data(item, indent + 1)
        print(f"{spacing}]")
    elif isinstance(data, tuple):
        print(f"{spacing}(")
        for item in data:
            print_data(item, indent + 1)
        print(f"{spacing})")
    else:
        print(f"{spacing}{repr(data)}")
    

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Read a3d print a pickle file.')
    parser.add_argument('--file_path', default="./actor_0/traj_5.pickle" ,type=str, help='Path to the pickle file.')
    args = parser.parse_args()

    # Read the pickle file
    data = read_pickle_file(args.file_path)

    # Print the data
    print("Contents of the pickle file:")
    print_data(data)
    print_data(np.sum(abs(data[-1][0][0])))

if __name__ == "__main__":
    main()
