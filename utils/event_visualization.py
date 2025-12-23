import pickle
import numpy as np
import matplotlib.pyplot as plt

def plot_event_heatmap(file_path, event_index=5):
    """
    Loads event data from a pickle file and plots a heatmap of a specified 16x16 event matrix.
    
    Parameters:
    - file_path (str): Path to the pickle file
    - event_index (int): Index of the event matrix to plot (default is 0)
    """
    try:
        # Load data
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        # print(data[1][0][0][0])
        # Extract the specified event's matrix and reshape it to 16x16 if possible
        event_matrix = data[event_index][0][0][0]
        if event_matrix.shape[0] >= 256:  # Check if it has enough data to reshape
            event_matrix_16x16 = event_matrix[:256].reshape(16, 16)
        else:
            raise ValueError("Event matrix does not contain enough data for 16x16 reshaping.")
        
        # Plot heatmap
        plt.figure(figsize=(6, 6))
        plt.imshow(event_matrix_16x16, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Event Intensity')
        plt.title(f'Heatmap of Event {event_index}')
        plt.show()
        
    except Exception as e:
        print("An error occurred:", e)

# Example usage
file_path = 'local_buffer/actor_0/traj_2.pickle'
data = plot_event_heatmap(file_path)
