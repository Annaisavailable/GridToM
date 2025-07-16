import os
import cv2
import json
import torch
import numpy as np

def get_key_frames(nodes):
    return [nodes[0], (nodes[0]+nodes[1])//2, nodes[1], (nodes[1]+nodes[2])//2, nodes[2], (nodes[2]+nodes[3])//2, nodes[3]]

def read_video_pyav(args, container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''

    image_process_mode = args.image_process_mode
    if image_process_mode != "Default":
        target_width = getattr(args, 'target_width', 224)
        target_height = getattr(args, 'target_height', 320)
        frames = []
        container.seek(0)
        start_index = indices[0]
        end_index = indices[-1]
        for i, frame in enumerate(container.decode(video=0)):
            if i > end_index:
                break
            if i >= start_index and i in indices:
                frame_array = frame.to_ndarray(format="rgb24")
                resized_frame = cv2.resize(
                    frame_array, 
                    (target_width, target_height), 
                    interpolation=cv2.INTER_LINEAR
                )
                frames.append(resized_frame)
        return np.stack(frames)
    else:
        frames = []
        container.seek(0)
        start_index = indices[0]
        end_index = indices[-1]
        for i, frame in enumerate(container.decode(video=0)):
            if i > end_index:
                break
            if i >= start_index and i in indices:
                frames.append(frame)
        return np.stack([x.to_ndarray(format="rgb24") for x in frames])

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def colorize(text, color):
    colors = {
        "blue": "\033[94m",
        "green": "\033[92m",
        "red": "\033[91m",
        "yellow": "\033[93m",
        "end": "\033[0m",
    }
    return colors[color] + text + colors["end"]

def print_colored(text, color):
    print(colorize(text, color))

def load_json(file_path):
    annotations = []
    with open(file_path, 'r') as file:
        for key, line in enumerate(file):
            data = json.loads(line)
            annotations.append(data)
    return annotations

def normalize_vectors(arr):
    """
    Normalizes the last dimension of a numpy array
    """
    # Calculate the norm (magnitude) of each (128,) vector
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)

    # Avoid division by zero by setting zero norms to one
    norms[norms == 0] = 1

    # Normalize the vectors
    normalized_arr = arr / norms
    return normalized_arr

def find_largest_k_items(arr, k):
    """
    Find the largest k items in a numpy array.

    Parameters:
    arr (numpy.array): A numpy array from which to find the largest k items.
    k (int): The number of largest items to find.

    Returns:
    list of tuples: A list of tuples, each containing the index and value of one of the k largest items.
    """
    # Flatten the array and get the indices of the largest k values
    indices = np.unravel_index(np.argsort(arr.ravel())[-k:], arr.shape)
    
    # Zip the indices together and get the corresponding values
    largest_items = [(index, arr[index]) for index in zip(*indices)]
    return largest_items[::-1]

def get_interventions_dict(all_activations, top_heads, directions):
    device = "cuda"
    directions = normalize_vectors(directions)
    interventions_dict = {}
    for (layer, head), val_acc in top_heads:
        dir = directions[layer, head]                     # Assuming normalized
        activations = all_activations[:, layer, head, :]  # N x 128
        proj_vals = activations @ dir.T
        proj_val_std = np.std(proj_vals)
    
        # Check if the layer key exists in the dictionary, if not, initialize an empty list
        if layer not in interventions_dict:
            interventions_dict[layer] = []
        dir = torch.tensor(dir).to(device)
        # Append the tuple (head, val_acc, proj_val_std) to the list corresponding to the layer
        interventions_dict[layer].append((head, dir, proj_val_std, val_acc))
    return interventions_dict