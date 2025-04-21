"""
Script to evaluate OpenCV's essential matrix estimation and pose recovery 
against COLMAP ground truth data.
"""

import os
import sys
import numpy as np
import cv2
import argparse
from tqdm import tqdm
import itertools
import scipy.linalg

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the colmap reading utilities
import colmap_read
import colmap_utils

print(cv2.__version__)


def rotation_error(R_est, R_gt):
    """
    Calculate rotation error: norm(Log(Rest^T*R_gt))
    
    Args:
        R_est: Estimated rotation matrix
        R_gt: Ground truth rotation matrix
        
    Returns:
        float: Rotation error
    """
    # R_rel = R_est.T @ R_gt
    R_rel = np.dot(R_est.T, R_gt)
    
    # Calculate the logarithm of the rotation matrix
    # This gives us the rotation vector
    rvec = cv2.Rodrigues(R_rel)[0]
    return np.linalg.norm(rvec)


def translation_error(t_est, t_gt):
    """
    Calculate translation error: 1-t_est.dot(t_gt) (cosine distance)
    
    Args:
        t_est: Estimated translation vector (normalized)
        t_gt: Ground truth translation vector (normalized)
        
    Returns:
        float: Translation error (0 means perfect alignment, 2 means opposite directions)
    """
    # Ensure the vectors are normalized
    t_est_norm = t_est / np.linalg.norm(t_est)
    t_gt_norm = t_gt / np.linalg.norm(t_gt)
    
    # Calculate cosine distance
    cos_dist = 1 - np.dot(t_est_norm, t_gt_norm)
    
    # Convert to degrees (angle between vectors)
    angle_deg = np.arccos(max(-1, min(1, 1 - cos_dist))) * 180 / np.pi
    
    return cos_dist


def get_relative_pose(image1, image2, cameras):
    """
    Compute the relative pose between two cameras.
    
    Args:
        image1: First image data from COLMAP
        image2: Second image data from COLMAP
        cameras: Dictionary of camera data
        
    Returns:
        tuple: (R, t) relative rotation and translation
    """
    # Get rotation matrices
    R1 = image1.qvec2rotmat()
    R2 = image2.qvec2rotmat()
    
    # Get translation vectors
    t1 = image1.tvec
    t2 = image2.tvec
    
    # T1 = np.eye(4)
    # T1[:3, :3] = R1
    # T1[:3, 3] = t1
    # T2 = np.eye(4)
    # T2[:3, :3] = R2
    # T2[:3, 3] = t2
    
    # T1 = np.linalg.inv(T1)
    # T2 = np.linalg.inv(T2)
    
    # R1 = T1[:3, :3].T
    # R_rel =(R1 @ T2[:3, :3]).T
    # t_rel =-R_rel@R1@(T2[:3, 3] - T1[:3, 3])
    
    # Compute relative rotation: R2 * R1^T
    R_rel = np.dot(R2, R1.T)  # This line is no longer needed
    # Compute relative translation: t2 - R_rel * t1  # This line is also no longer needed
    t_rel = t2 - np.dot(R_rel, t1)  # This line is also no longer needed
    
    # Normalize translation
    t_rel = t_rel / np.linalg.norm(t_rel)
    
    return R_rel, t_rel


def find_matching_points(image1, image2, points3D):
    """
    Find matching points between two images based on COLMAP data.
    
    Args:
        image1: First image data from COLMAP
        image2: Second image data from COLMAP
        points3D: 3D points data from COLMAP
        
    Returns:
        tuple: (points1, points2) corresponding points in each image
    """
    # Find common 3D points visible in both images
    common_point3D_ids = set(image1.xys.keys()) & set(image2.xys.keys())
    
    if len(common_point3D_ids) < 100:
        return None, None
    
    # Extract matching 2D points
    points1 = []
    points2 = []
    
    for point3D_id in common_point3D_ids:
        points1.append(image1.xys[point3D_id])
        points2.append(image2.xys[point3D_id])
    
    return np.array(points1), np.array(points2)


def evaluate_pose_estimation(colmap_path, min_translation=0.075):
    """
    Evaluate OpenCV's essential matrix estimation and pose recovery.
    
    Args:
        colmap_path: Path to COLMAP data
        min_translation: Minimum translation magnitude to consider a pair
        
    Returns:
        dict: Evaluation results
    """
    # Read COLMAP data
    print(f"Reading COLMAP data from {colmap_path}...")
    cameras = colmap_read.read_cameras_text(os.path.join(colmap_path, "cameras.txt"))
    images = colmap_read.read_images_text(os.path.join(colmap_path, "images.txt"))
    points3D = colmap_read.read_points3D_text(os.path.join(colmap_path, "points3D.txt"))
    
    # Results container
    results = {
        'rotation_errors': [],
        'translation_errors': [],
        'num_inliers': [],
        'pair_info': []
    }
    
    # Get all image pairs
    image_ids = list(images.keys())
    total_pairs = 0
    valid_pairs = 0
    
    print("Evaluating image pairs...")
    for id1, id2 in tqdm(itertools.combinations(image_ids, 2), total=len(image_ids)*(len(image_ids)-1)//2):
        image1 = images[id1]
        image2 = images[id2]
        
        # Get ground truth relative pose
        R_gt, t_gt = get_relative_pose(image1, image2, cameras)
        
        # Check if translation magnitude is large enough
        translation_magnitude = np.linalg.norm(image1.tvec - image2.tvec)
        if translation_magnitude < min_translation:
            continue
        
        total_pairs += 1
        
        # Find matching points
        points1, points2 = find_matching_points(image1, image2, points3D)
        if points1 is None or len(points1) < 8:
            continue
        
        # Get camera intrinsics
        K1 = colmap_utils.get_cam(image1, cameras)
        K2 = colmap_utils.get_cam(image2, cameras)
        
        # OpenCV essential matrix estimation
        E, mask = cv2.findEssentialMat(
            points1, points2, K1, method=cv2.LMEDS, prob=0.999, threshold=1.0, maxIters=2000
        )
        
        if E is None or E.shape != (3, 3):
            continue
            
        # Count inliers
        num_inliers = np.sum(mask)
        
        # Recover pose
        _, R_est, t_est, mask_pose = cv2.recoverPose(E, points1, points2, K1, mask=mask)
        
        # Calculate errors
        rot_error = rotation_error(R_est, R_gt)
        trans_error = translation_error(t_est.flatten(), t_gt)
        
        # Store results
        if not np.isinf(rot_error):
            results['rotation_errors'].append(rot_error)
            results['translation_errors'].append(trans_error)
            results['num_inliers'].append(num_inliers)
            results['pair_info'].append((image1.name, image2.name, translation_magnitude))
            valid_pairs += 1
    
    # Compute statistics
    if valid_pairs > 0:
        results['mean_rotation_error'] = np.mean(results['rotation_errors'])
        results['median_rotation_error'] = np.median(results['rotation_errors'])
        results['mean_translation_error'] = np.mean(results['translation_errors'])
        results['median_translation_error'] = np.median(results['translation_errors'])
    
    print(f"Evaluated {total_pairs} pairs with translation > {min_translation}")
    print(f"Found {valid_pairs} valid pairs with enough matching points")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate OpenCV's essential matrix estimation")
    parser.add_argument('--colmap_path', required=True, 
                       help='Path to COLMAP data folder containing cameras.txt, images.txt, etc.')
    parser.add_argument('--min_translation', type=float, default=0.075,
                       help='Minimum translation magnitude to consider a pair (default: 0.075)')
    args = parser.parse_args()
    
    results = evaluate_pose_estimation(args.colmap_path, args.min_translation)
    
    # Print results
    if 'mean_rotation_error' in results:
        print("\n=== Evaluation Results ===")
        print(f"Number of evaluated pairs: {len(results['rotation_errors'])}")
        print(f"Mean rotation error: {results['mean_rotation_error']:.2f} degrees")
        print(f"Median rotation error: {results['median_rotation_error']:.2f} degrees")
        print(f"Mean translation error: {results['mean_translation_error']:.2f} degrees")
        print(f"Median translation error: {results['median_translation_error']:.2f} degrees")
        
        # Print some example pairs
        print("\n=== Sample Pairs ===")
        num_samples = min(5, len(results['pair_info']))
        for i in range(num_samples):
            img1, img2, trans_mag = results['pair_info'][i]
            print(f"Pair {i+1}: {os.path.basename(img1)} - {os.path.basename(img2)}")
            print(f"  Translation magnitude: {trans_mag:.3f}")
            print(f"  Rotation error: {results['rotation_errors'][i]:.2f} degrees")
            print(f"  Translation error: {results['translation_errors'][i]:.2f} degrees")
            print(f"  Inliers: {results['num_inliers'][i]}")
    else:
        print("No valid pairs found for evaluation.")


if __name__ == "__main__":
    main()
