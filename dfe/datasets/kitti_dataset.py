"""KITTI dataset for fundamental matrix estimation. Derived from FundamentalMatrixDataset.
"""

import numpy as np
import os
from tqdm import tqdm

from dfe.datasets import FundamentalMatrixDataset


class KITTIDataset(FundamentalMatrixDataset):
    """KITTI dataset for fundamental matrix estimation. Derived from FundamentalMatrixDataset.
    """

    def __init__(
        self,
        path,
        sequence,
        detector="sift",
        threshold=1.0,
        confidence=0.999,
        num_points=-1,
        max_F=None,
        random=False,
        min_matches=100,
        compute_virtual_points=True,
        visualize=False,
        vis_output_dir=None,
        mode="test"  # Added mode parameter, default is "test"
    ):
        """Init.

        Args:
            path (str): path to dataset root folder
            sequence (str or int): sequence number (e.g., "00", "01", etc.)
            detector (str, optional): feature detector type. Defaults to "sift".
            threshold (float, optional): epipolar threshold used. Defaults to 1.0.
            confidence (float, optional): RANSAC confidence used. Defaults to 0.999.
            num_points (int, optional): number of points per sample. Defaults to -1.
            max_F (int, optional): maximal number of samples (if None: use all). Defaults to None.
            random (bool, optional): random database access. Defaults to False.
            min_matches (int, optional): minimal number of good matches per sample. Defaults to 20.
            compute_virtual_points (bool, optional): whether to compute virtual points. Defaults to True.
            visualize (bool, optional): whether to visualize matches. Defaults to True.
            vis_output_dir (str, optional): directory to save visualizations. If None, uses 'vis' under dataset path. Defaults to None.
            mode (str, optional): "train" or "test" mode. Defaults to "test".
        """
        super(KITTIDataset, self).__init__(num_points)
        
        self.mode = mode
        self.compute_virtual_points = compute_virtual_points
        
        # Ensure sequence is a string with two digits
        if isinstance(sequence, int):
            sequence = f"{sequence:02d}"
        elif isinstance(sequence, str) and len(sequence) == 1:
            sequence = f"0{sequence}"
            
        print(f"Loading KITTI sequence {sequence}")
        
        # Setup paths for dataset folders
        dataset_root = os.path.dirname(path)  # Get the parent directory of path's parent
        sequence_dir = path
        eval_dir = os.path.join(sequence_dir, "2_99_sift")
        matches_file = os.path.join(eval_dir, "matches.txt")
        
        # Path to the precomputed relative pose file
        pose_file = os.path.join(dataset_root, "poses", f"{sequence}_rel.txt")
        print(f"Using relative pose file: {pose_file}")
        
        # Load camera calibration
        calib_file = os.path.join(sequence_dir, "calib.txt")
        K = self._load_calibration(calib_file)
        
        # Load the precomputed matches and poses
        matches_data = self._load_matches(matches_file, min_matches)
        poses_data = self._load_poses(pose_file)
        
        # Initialize data containers
        self.img_paths = []
        self.pts = []
        self.R_gt = []  # Ground truth rotation matrix
        self.t_gt = []  # Ground truth translation vector (normalized)
        self.intrinsics = []
        
        # Process matches and poses
        print(f"Processing {len(matches_data)} image pairs...")
        processed_pairs = 0
        max_pairs = max_F if max_F else float('inf')
        
        for i, (img1_idx, img2_idx, pts1, pts2) in enumerate(tqdm(matches_data)):
            if processed_pairs >= max_pairs:
                break
                
            # Skip if indices don't match
            if int(img1_idx) >= len(poses_data) or int(img2_idx) != int(img1_idx) + 1:
                continue
            
            # Get relative pose
            pose = poses_data[int(img1_idx)]
            R_rel = pose[:3, :3]
            t_rel = pose[:3, 3]
            
            # Normalize t_rel to unit vector if it's not already normalized
            t_norm = np.linalg.norm(t_rel)
            if t_norm > 0.5 :  # Avoid division by zero
                t_rel_norm = t_rel / t_norm
            else:
                continue
            
            # Create pairs with point coordinates
            pairs = np.zeros((len(pts1), 7))  # [x1, y1, x2, y2, 0, 0, 0] (adding placeholders for side info)
            pairs[:, 0:2] = pts1
            pairs[:, 2:4] = pts2
            
            # Proceed only if we have enough matches
            if len(pairs) >= min_matches:
                self.pts.append(pairs)
                self.R_gt.append(R_rel)
                self.t_gt.append(t_rel_norm)
                self.intrinsics.append(K)
                
                img1_path = os.path.join(sequence_dir, "image_0", f"{img1_idx}.png")
                img2_path = os.path.join(sequence_dir, "image_0", f"{img2_idx}.png")
                self.img_paths.append((img1_path, img2_path))
                
                processed_pairs += 1
                
        
        print(f"Created {len(self.pts)} pairs for training/evaluation")
        
        # Precompute F_gt and virtual points for training mode
        if mode == "train" and compute_virtual_points:
            self.F_gt = []
            self.pts1_virt = []
            self.pts2_virt = []
            
            for i in range(len(self.pts)):
                # Calculate fundamental matrix from R, t, and K
                R = self.R_gt[i]
                t = self.t_gt[i]
                K_mat = self.intrinsics[i]
                
                # Calculate essential matrix
                t_cross = np.array([
                    [0, -t[2], t[1]],
                    [t[2], 0, -t[0]],
                    [-t[1], t[0], 0]
                ])
                E = t_cross @ R
                
                # Calculate fundamental matrix
                F = np.linalg.inv(K_mat).T @ E @ np.linalg.inv(K_mat)
                F = F / np.linalg.norm(F)  # Normalize F
                self.F_gt.append(F)
                
                # Generate virtual points for training
                if compute_virtual_points:
                    # Generate virtual points covering the image
                    h, w = 376, 1241  # KITTI image size
                    pts1_virt = np.zeros((100, 3))
                    pts2_virt = np.zeros((100, 3))
                    
                    # Create grid of points
                    for j in range(10):
                        for k in range(10):
                            idx = j * 10 + k
                            x = w * (k + 0.5) / 10
                            y = h * (j + 0.5) / 10
                            pts1_virt[idx] = [x, y, 1]
                            pts2_virt[idx] = [x, y, 1]
                    
                    self.pts1_virt.append(pts1_virt)
                    self.pts2_virt.append(pts2_virt)

    def _load_calibration(self, calib_file):
        """Load camera calibration matrix from KITTI calibration file.
        
        Args:
            calib_file (str): Path to the calibration file
            
        Returns:
            np.ndarray: 3x3 camera intrinsic matrix
        """
        K = np.zeros((3, 3))
        K[0, 0] = 718.856
        K[0, 2] = 607.1928
        K[1, 1] = 718.856
        K[1, 2] = 185.2157
        K[2, 2] = 1.0
        return K

    def _load_matches(self, matches_file, min_matches=20):
        """Load precomputed matches from matches.txt file.
        
        Args:
            matches_file (str): Path to the matches file
            min_matches (int): Minimum number of matches to consider an image pair
            
        Returns:
            list: List of tuples (img1_idx, img2_idx, pts1, pts2)
        """
        matches_data = []
        
        try:
            with open(matches_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 3:
                        # Extract image indices
                        img_indices = parts[0].strip().split()
                        if len(img_indices) != 2:
                            continue
                            
                        img1_idx, img2_idx = img_indices
                        
                        # Extract point coordinates
                        pts1_str = parts[1].strip().split()
                        pts2_str = parts[2].strip().split()
                        
                        # Parse points
                        pts1 = []
                        pts2 = []
                        for i in range(0, len(pts1_str), 2):
                            if i+1 < len(pts1_str):
                                pts1.append([float(pts1_str[i]), float(pts1_str[i+1])])
                                
                        for i in range(0, len(pts2_str), 2):
                            if i+1 < len(pts2_str):
                                pts2.append([float(pts2_str[i]), float(pts2_str[i+1])])
                        
                        # Convert to numpy arrays
                        pts1 = np.array(pts1)
                        pts2 = np.array(pts2)
                        
                        # Only add if we have enough matches
                        if len(pts1) >= min_matches and len(pts1) == len(pts2):
                            matches_data.append((img1_idx, img2_idx, pts1, pts2))
            
            print(f"Loaded {len(matches_data)} image pairs from {matches_file}")
            return matches_data
            
        except Exception as e:
            print(f"Error loading matches from {matches_file}: {e}")
            return []

    def _load_poses(self, pose_file):
        """Load ground truth poses from pose file.
        
        Args:
            pose_file (str): Path to the pose file
            
        Returns:
            list: List of 3x4 transformation matrices
        """
        try:
            poses = np.loadtxt(pose_file).reshape(-1, 3, 4)
            print(f"Loaded {len(poses)} relative poses from {pose_file}")
            return poses
        except Exception as e:
            print(f"Error loading poses from {pose_file}: {e}")
            return []

    def __getitem__(self, index):
        """Get dataset sample.

        Args:
            index (int): sample index

        Returns:
            If mode is "train":
                tuple: points, side information, ground truth fundamental matrix, virtual points 1, virtual points 2
            If mode is "test":
                tuple: points, side information, ground truth rotation and translation, camera matrix
        """
        pts = self.pts[index]
        
        # add data if too small for training
        if self.num_points > 0 and pts.shape[0] < self.num_points:
            while pts.shape[0] < self.num_points:
                num_missing = self.num_points - pts.shape[0]
                idx = np.random.permutation(pts.shape[0])[:num_missing]

                pts_pert = pts[idx]
                pts = np.concatenate((pts, pts_pert), 0)

        # normalize side information
        side_info = pts[:, 4:] / np.maximum(np.amax(pts[:, 4:], 0), 1e-10)

        # Get point coordinates
        pts = pts[:, :4]

        # remove data if too big for training
        if self.num_points > 0 and (pts.shape[0] > self.num_points):
            idx = np.random.permutation(pts.shape[0])[: self.num_points]

            pts = pts[idx, :]
            side_info = side_info[idx]

        if self.mode == "train":
            # Return data for training
            F_gt = self.F_gt[index]
            pts1_virt = self.pts1_virt[index]
            pts2_virt = self.pts2_virt[index]
            return (np.float32(pts), side_info, np.float32(F_gt), np.float32(pts1_virt), np.float32(pts2_virt))
        else:
            # Return data for testing
            R_gt = self.R_gt[index]
            t_gt = self.t_gt[index]
            K = self.intrinsics[index]
            return (np.float32(pts), side_info, np.float32(R_gt), np.float32(t_gt), np.float32(K))
    
    def __len__(self):
        """Get length of dataset.

        Returns:
            int: length
        """
        return len(self.R_gt)


