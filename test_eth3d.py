"""Script for testing the NormalizedEightPointNet.

    to see help:
    $ python test.py -h

"""

import argparse
import cv2
import os
import numpy as np
import torch
from tqdm import tqdm

from dfe.datasets import ETH3D  # Update the import
from dfe.models import NormalizedEightPointNet
from dfe.utils import compute_residual


def save_err(dataset_name, R_err, t_err):
    """
    Save rotation and translation errors to text files.
    Also saves accumulated error statistics.
    
    Args:
        dataset_name (str): Dataset name to use as prefix for output files
        R_err (list): List of rotation errors
        t_err (list): List of translation errors
    """
    output_dir = "/home/neo/Epipolar_evaluation/output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Use dataset name as prefix
    prefix = f"{dataset_name}_"
    
    # Save individual errors
    with open(os.path.join(output_dir, f"{prefix}errors.txt"), "w") as f:
        for r_err, t_err_val in zip(R_err, t_err):
            f.write(f"{r_err},{t_err_val}\n")
    
    # Save accumulated error statistics
    with open(os.path.join(output_dir, f"{prefix}error_stats.txt"), "w") as f:
        r_err_array = np.array(R_err)
        t_err_array = np.array(t_err)
        
        # Calculate statistics
        r_err_mean = np.mean(r_err_array)
        r_err_median = np.median(r_err_array)
        r_err_std = np.std(r_err_array)
        r_err_sum = np.sum(r_err_array)
        r_max = np.max(r_err_array)
        
        t_err_mean = np.mean(t_err_array)
        t_err_median = np.median(t_err_array)
        t_err_std = np.std(t_err_array)
        t_err_sum = np.sum(t_err_array)
        t_max = np.max(t_err_array)
        
        # Write statistics
        f.write(f"Rotation Error (radian):\n")
        f.write(f"Mean: {r_err_mean}\n")
        f.write(f"Median: {r_err_median}\n")
        f.write(f"Std Dev: {r_err_std}\n")
        f.write(f"Sum: {r_err_sum}\n")
        f.write(f"Max: {r_max}\n\n")
        
        f.write(f"Translation Error (cosine distance):\n")
        f.write(f"Mean: {t_err_mean}\n")
        f.write(f"Median: {t_err_median}\n")
        f.write(f"Std Dev: {t_err_std}\n")
        f.write(f"Sum: {t_err_sum}\n")
        f.write(f"Max: {t_max}\n")
    
    print(f"Errors saved to {output_dir}/{prefix}errors.txt and {output_dir}/{prefix}error_stats.txt")


def eval_model(pts, side_info, model, device, postprocess=True):
    pts_orig = pts.copy()
    pts = torch.from_numpy(pts).to(device).unsqueeze(0)
    side_info = torch.from_numpy(side_info).to(torch.float).to(device).unsqueeze(0)

    F_est, rescaling_1, rescaling_2, weights = model(pts, side_info)

    F_est = rescaling_1.permute(0, 2, 1).bmm(F_est[-1].bmm(rescaling_2))

    F_est = F_est / F_est[:, -1, -1].unsqueeze(-1).unsqueeze(-1)
    F = F_est[0].data.cpu().numpy()
    weights = weights[0, 0].data.cpu().numpy()

    F_best = F
    return F_best


def test(options):
    """Test NormalizedEightPointNet.

    Args:
        options: testing options
    """
    device = torch.device("cpu")

    print("device: %s" % device)

    print("-- Data loading --")
    
    model = NormalizedEightPointNet(
            depth=options.depth, side_info_size=options.side_info_size
        )

    model.load_state_dict(torch.load(options.model, map_location='cpu'))
    model.to(device)
    model = model.eval()
    
    # Process each dataset separately to keep track of their errors individually
    for dset_path in options.dataset:
        # Extract the dataset name from the path (last part)
        dataset_name = os.path.basename(os.path.normpath(dset_path))
        print(f'Processing dataset "{dataset_name}" from "{dset_path}"')
        
        # Create a new dataset with mode="test"
        dataset = ETH3D(
            dset_path,
            num_points=-1,
            compute_virtual_points=False,
            max_F=10000,
            random=False,
            mode="test"  # Set mode to "test"
        )
        
        print(f"Dataset size: {len(dataset)}")

        if len(dataset) == 0:
            print(f"Warning: Dataset {dataset_name} is empty. Skipping.")
            continue

        # Per-dataset error tracking
        R_err_dataset = []
        t_err_dataset = []
        
        # Process all samples in this dataset
        idxs = len(dataset)

        for idx in tqdm(range(idxs), desc=f"Processing {dataset_name}"):
            (pts, side_info, R_gt, t_gt, K) = dataset[idx]

            # Compute our result
            with torch.no_grad():
                F_ours = eval_model(pts, side_info, model, device)

                pts1 = pts[:, :2]
                pts2 = pts[:, 2:4]
                K = K.astype(np.float64)
                F_ours = F_ours.astype(np.float64)
                
                # Calculate Essential matrix from F and K
                E_ours = K.T @ F_ours @ K
                
                try:
                    # Recover pose from essential matrix
                    _, R_est, t_est, _ = cv2.recoverPose(E_ours, pts1, pts2, K)
                    
                    # Calculate rotation error with proper bounds to prevent NaN
                    R_rel = R_est @ R_gt
                    # Clip to valid range for arccos to prevent NaN
                    cos_angle = np.clip((np.trace(R_rel) - 1) / 2.0, -1.0, 1.0)
                    rot_err_deg = np.arccos(cos_angle)
                    rot_err_deg = min(np.degrees(rot_err_deg), 180 - np.degrees(rot_err_deg)) / 180*np.pi
                    R_err_dataset.append(rot_err_deg)
                    
                    # Calculate translation error (cosine distance)
                    # Ensure vectors are normalized before calculating dot product
                    t_est_norm = t_est.flatten() / (np.linalg.norm(t_est) + 1e-10)
                    t_gt_norm = t_gt.flatten() / (np.linalg.norm(t_gt) + 1e-10)
                    cos_dist = 1.0 - np.abs(np.dot(t_gt_norm, -R_gt@t_est_norm))
                    t_err_dataset.append(cos_dist)
                except Exception as e:
                    print(f"Error processing sample {idx} in dataset {dataset_name}: {e}")
                    continue
        # Save errors for this dataset
        if R_err_dataset and t_err_dataset:
            save_err(dataset_name, R_err_dataset, t_err_dataset)
        else:
            print(f"No valid results for dataset {dataset_name}")


if __name__ == "__main__":
    np.random.seed(42)
    PARSER = argparse.ArgumentParser(description="Testing")

    PARSER.add_argument("--depth", type=int, default=3, help="depth")
    PARSER.add_argument(
        "--side_info_size", type=int, default=3, help="size of side information"
    )
    PARSER.add_argument(
        "--dataset", default=[
                            "/home/neo/Epipolar_evaluation/dataset/botanical_garden"
                           ,"/home/neo/Epipolar_evaluation/dataset/statue"
                           ,"/home/neo/Epipolar_evaluation/dataset/bridge"
                           ,"/home/neo/Epipolar_evaluation/dataset/door"
                           ,"/home/neo/Epipolar_evaluation/dataset/old_computer"
                           ,"/home/neo/Epipolar_evaluation/dataset/exhibition_hall"
                           ,"/home/neo/Epipolar_evaluation/dataset/observatory"
                           ,"/home/neo/Epipolar_evaluation/dataset/lounge"
                           ,"/home/neo/Epipolar_evaluation/dataset/boulders"
                           ,"/home/neo/Epipolar_evaluation/dataset/lecture_room"
                           ,"/home/neo/Epipolar_evaluation/dataset/terrace_2"], nargs="+", help="list of datasets"
    )
    PARSER.add_argument("--num_workers", type=int, default=1, help="number of workers")
    PARSER.add_argument("--model", type=str, default="/home/neo/Epipolar_evaluation/dl/DFE/models/eth3d.pt", help="model file")

    ARGS = PARSER.parse_args()

    test(ARGS)