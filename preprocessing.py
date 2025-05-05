import numpy as np
import pyvista as pv
import os
from scipy.spatial.transform import Rotation as R



def load_landmarks(landmark_path):
    """Load landmarks from an NPZ file."""
    if not landmark_path.endswith(".npz"):
        raise ValueError("Landmarks must be in NPZ format.")
    data = np.load(landmark_path)
    return data["arr_0"]


def save_to_npz(output_path, labels, vertices, landmarks):

    """Save landmarks, vertices, and heatmaps to an NPZ file."""
    np.savez(output_path, labels=labels, vertices=vertices, landmarks=landmarks)


def augment_geometry(mesh, landmarks, scale_range=(0.9, 1.1), rotation_range=(-10, 10), translation_range=(-5, 5), flip_axes=('x', 'y', 'z')):
    """Apply augmentation transformations including scaling, rotation, translation, and flipping."""
    # Scaling
    scale_factor = np.random.uniform(*scale_range)
    mesh.points *= scale_factor  # Scale the mesh points directly
    landmarks *= scale_factor

    # Rotation
    angle = np.radians(np.random.uniform(*rotation_range))
    axis = np.random.choice(['x', 'y', 'z'])
    rotation_vector = {
        'x': [angle, 0, 0],
        'y': [0, angle, 0],
        'z': [0, 0, angle]
    }[axis]
    rotation = R.from_rotvec(rotation_vector)
    rotation_matrix = rotation.as_matrix()

    # Apply rotation to the mesh
    mesh.points = (rotation_matrix @ mesh.points.T).T

    # Apply rotation to the landmarks
    landmarks = (rotation_matrix @ landmarks.T).T

    # Translation
    translation_vector = np.random.uniform(*translation_range, size=3)
    mesh.points += translation_vector
    landmarks += translation_vector

    # Flipping (using reflection matrices)
    flip_axis = np.random.choice(flip_axes)  # Randomly choose an axis to flip
    if flip_axis == 'x':
        mesh.points[:, 0] *= -1
        landmarks[:, 0] *= -1
    elif flip_axis == 'y':
        mesh.points[:, 1] *= -1
        landmarks[:, 1] *= -1
    elif flip_axis == 'z':  # Flip upside down
        mesh.points[:, 2] *= -1
        landmarks[:, 2] *= -1

    return mesh, landmarks


def compute_heatmaps(vert, landmarks, sigma=10):
    shape_sample = vert.reshape(vert.shape[0], 1, vert.shape[1]).repeat(landmarks.shape[0], axis=1)
    Euclidean_distance_i = np.linalg.norm((shape_sample - landmarks), axis=2)
    D2 = Euclidean_distance_i * Euclidean_distance_i
    S2 = 2.0 * sigma * sigma
    Exponent = D2 / S2
    heatmap = np.exp(-Exponent)
    return heatmap



def add_noise_to_point_cloud(point_cloud, noise_type='gaussian', noise_level=0.01):
    """
    Adds noise to a 3D point cloud.

    Parameters:
        point_cloud (np.ndarray): A numpy array of shape (N, 3) representing the point cloud.
        noise_type (str): Type of noise to add ('gaussian' or 'uniform').
        noise_level (float): Standard deviation of the Gaussian noise or range of the uniform noise.

    Returns:
        np.ndarray: The point cloud with added noise, same shape as input.
    """
    if not isinstance(point_cloud, np.ndarray):
        raise ValueError("Point cloud must be a numpy array.")

    if point_cloud.shape[1] != 3:
        raise ValueError("Point cloud must have shape (N, 3).")

    if noise_type == 'gaussian':
        noise = np.random.normal(loc=0.0, scale=noise_level, size=point_cloud.shape)
    elif noise_type == 'uniform':
        noise = np.random.uniform(low=-noise_level, high=noise_level, size=point_cloud.shape)
    else:
        raise ValueError("Invalid noise type. Choose 'gaussian' or 'uniform'.")

    noisy_point_cloud = point_cloud + noise
    return noisy_point_cloud





def process_and_augment_stls(stl_dir, landmark_dir, output_dir):
    """Process STL files, augment if needed, and compute heatmaps."""
    os.makedirs(os.path.join(output_dir,"1"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "stls"), exist_ok=True)

    for stl_file in os.listdir(stl_dir):
        if not stl_file.endswith(".stl"):
            continue

        base_name = os.path.splitext(stl_file)[0]
        landmark_file = os.path.join(landmark_dir, f"{base_name}.npz")

        if not os.path.exists(landmark_file):
            print(f"Landmarks for {stl_file} not found. Skipping.")
            continue

        # Load STL and landmarks
        mesh = pv.read(os.path.join(stl_dir, stl_file))
        landmarks = load_landmarks(landmark_file)

        if "USR" in base_name:
            # Augment 10 times
            for i in range(10):
                augmented_mesh, augmented_landmarks = augment_geometry(mesh.copy(), landmarks.copy())

                # Extract vertices
                vertices = augmented_mesh.points

                # Compute heatmaps
                heatmaps = compute_heatmaps(vertices, augmented_landmarks)
                labelmap = np.zeros((vertices.shape[0]))
                for idx, heatmap in enumerate(heatmaps.transpose(1, 0)):
                    ind = np.where(heatmap >= 0.7 * np.max(heatmap))
                    labelmap[ind[0]] = idx+1



                noisy_point_cloud = add_noise_to_point_cloud(vertices, noise_type='gaussian', noise_level=0.5)

                # Save augmented landmarks, vertices, and heatmaps in NPZ format
                output_npz_path = os.path.join(output_dir,"1", f"{base_name}_augmented_{i}.npz")
                save_to_npz(output_npz_path, labelmap, noisy_point_cloud, augmented_landmarks)


                # Optionally save the augmented STL
                augmented_mesh.save(os.path.join(output_dir, "stls", f"{base_name}_augmented_{i}.stl"))
        else:
            # Save without augmentation
            vertices = mesh.points
            heatmaps = compute_heatmaps(vertices, landmarks)
            labelmap = np.zeros((vertices.shape[0]))
            for idx, heatmap in enumerate(heatmaps.transpose(1, 0)):
                ind = np.where(heatmap >= 0.7 * np.max(heatmap))
                labelmap[ind[0]] = idx+1


            # Save landmarks, vertices, and heatmaps in NPZ format
            output_npz_path = os.path.join(output_dir,"1", f"{base_name}.npz")
            save_to_npz(output_npz_path, labelmap, vertices, landmarks)

            # Save the STL
            mesh.save(os.path.join(output_dir, "stls", f"{base_name}.stl"))

        print(f"Processed {stl_file}")


if __name__ == "__main__":
    stl_dir = "/Users/aidanamassalimova/Documents/PSP Planning Data/CT_based_dataset/Aidana_planning_based/stls"
    landmark_dir = "/Users/aidanamassalimova/Documents/PSP Planning Data/CT_based_dataset/Aidana_planning_based/landmarks"
    output_dir = "../new_new_dataset"

    # Process files
    process_and_augment_stls(stl_dir, landmark_dir, output_dir)



