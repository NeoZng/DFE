"""Colmap dataset for fundametal matrix estimation. Derived from FundamentalMatrixDataset.
"""

import numpy as np

from dfe.datasets import FundamentalMatrixDataset
from dfe.utils import colmap_read, colmap_utils


class ColmapDataset(FundamentalMatrixDataset):
    """Colmap dataset for fundametal matrix estimation. Derived from FundamentalMatrixDataset.
    """

    def __init__(
        self,
        path,
        num_points=-1,
        threshold=1,
        max_F=None,
        random=False,
        min_matches=20,
        compute_virtual_points=True,
    ):
        """Init.

        Args:
            path (str): path to dataset folder
            num_points (int, optional): number of points per sample. Defaults to -1.
            threshold (int, optional): epipolar threshold. Defaults to 1.
            max_F (int, optional): maximal number of samples (if None: use all). Defaults to None.
            random (bool, optional): random database access. Defaults to False.
            min_matches (int, optional): minimal number of good matches per sample. Defaults to 20.
        """
        super(ColmapDataset, self).__init__(num_points)
        cameras = colmap_read.read_cameras_text("%s/dslr_calibration_undistorted/cameras.txt" % path)
        images = colmap_read.read_images_text("%s/dslr_calibration_undistorted/images.txt" % path)

        self.img_paths = []
        self.pts = []
        self.R_gt = []  # Ground truth rotation matrix
        self.t_gt = []  # Ground truth translation vector (normalized)
        self.size_1 = []
        self.size_2 = []
        self.intrinsics = []

        # Traverse all image pairs
        img_ids = list(images.keys())
        min_covisible_points = 100  # Minimum number of covisible points
        min_translation_norm = 0.075  # Minimum translation norm threshold
        
        
        for row in cursor:
            # max number of image pairs
            if max_F and len(self.F) == max_F:
                break

            img1_id, img2_id = colmap_utils.pair_id_to_image_ids(row[0])

            try:
                img1 = images[img1_id]
                img2 = images[img2_id]
            except KeyError:
                print("Image doesn't match id")
                continue

            # check if both images share enough 3D points
            pts1 = img1.point3D_ids[img1.point3D_ids != -1]
            pts2 = img2.point3D_ids[img2.point3D_ids != -1]

            common = len(np.intersect1d(pts1, pts2))

            if common < min_matches:
                continue

            # get cameras
            K1, T1, sz1 = colmap_utils.get_cam(img1, cameras)
            K2, T2, sz2 = colmap_utils.get_cam(img2, cameras)

            F = colmap_utils.compose_fundamental_matrix(K1, T1, K2, T2)

            # pull the matches
            matches = np.fromstring(row[1], dtype=np.uint32).reshape(-1, 2)

            cursor_2 = connection.cursor()
            cursor_2.execute(
                "SELECT data, cols FROM keypoints WHERE image_id=?;", (img1_id,)
            )
            row_2 = next(cursor_2)
            keypoints1 = np.fromstring(row_2[0], dtype=np.float32).reshape(-1, row_2[1])

            cursor_2.execute(
                "SELECT data, cols FROM keypoints WHERE image_id=?;", (img2_id,)
            )
            row_2 = next(cursor_2)
            keypoints2 = np.fromstring(row_2[0], dtype=np.float32).reshape(-1, row_2[1])

            cursor_2.execute(
                "SELECT data FROM descriptors WHERE image_id=?;", (img1_id,)
            )
            row_2 = next(cursor_2)
            descriptor_1 = np.float32(
                np.fromstring(row_2[0], dtype=np.uint8).reshape(-1, 128)
            )

            cursor_2.execute(
                "SELECT data FROM descriptors WHERE image_id=?;", (img2_id,)
            )
            row_2 = next(cursor_2)
            descriptor_2 = np.float32(
                np.fromstring(row_2[0], dtype=np.uint8).reshape(-1, 128)
            )

            dist = np.sqrt(
                np.mean(
                    (descriptor_1[matches[:, 0]] - descriptor_2[matches[:, 1]]) ** 2, 1
                )
            )[..., None]

            rel_scale = np.abs(
                keypoints1[matches[:, 0], 2] - keypoints2[matches[:, 1], 2]
            )[..., None]

            angle1 = keypoints1[matches[:, 0], 3]
            angle2 = keypoints2[matches[:, 1], 3]

            rel_orient = np.minimum(np.abs(angle1 - angle2), np.abs(angle2 - angle1))[
                ..., None
            ]
            # rel_orient = np.abs(angle1 - angle2)[..., None]

            pairs = np.hstack(
                (
                    keypoints1[matches[:, 0], :2],
                    keypoints2[matches[:, 1], :2],
                    dist,
                    rel_scale,
                    rel_orient,
                )
            )
            dist = colmap_utils.compute_residual(pairs[:, :4], F.T)
            
            
        for i in range(len(img_ids)):
                
            img1_id = img_ids[i]
            img1 = images[img1_id]
            
            for j in range(i+1, len(img_ids)):
                img2_id = img_ids[j]
                img2 = images[img2_id]
                
                # Find common 3D points between the two images using the xys dictionary
                common_points = set(img1.xys.keys()).intersection(set(img2.xys.keys()))
                
                # Skip if not enough covisible points
                if len(common_points) < min_covisible_points:
                    continue
                
                # Get cameras
                K1 = colmap_utils.get_cam(img1, cameras)
                
                # Calculate relative pose
                R1 = img1.qvec2rotmat()
                R2 = img2.qvec2rotmat()
                t1 = img1.tvec
                t2 = img2.tvec
                R_rel = np.dot(R1.T, R2)
                t_rel = np.dot(R1.T, t2 - t1)
                
                # Skip pairs with small translation
                if np.linalg.norm(t_rel) < min_translation_norm:
                    continue
                
                # Normalize t_rel to unit vector
                t_rel_norm = t_rel / np.linalg.norm(t_rel)
                
                # Extract corresponding points
                pts1 = []
                pts2 = []
                
                for point3D_id in common_points:
                    xy1 = img1.xys[point3D_id]
                    xy2 = img2.xys[point3D_id]
                    pts1.append(xy1)
                    pts2.append(xy2)
                
                pts1 = np.array(pts1)
                pts2 = np.array(pts2)
                
                # Create pairs with dummy side information (for compatibility)
                pairs = np.hstack((
                    pts1,
                    pts2,
                    np.zeros((len(pts1), 3))  # Placeholder for side info: dist, rel_scale, rel_orient
                ))
                

                self.pts.append(pairs)
                self.R_gt.append(R_rel)
                self.t_gt.append(t_rel_norm)
                self.intrinsics.append(K1)
                
                img1_path = "%s/%s" % (path, img1.name)
                img2_path = "%s/%s" % (path, img2.name)
                self.img_paths.append((img1_path, img2_path))


    def __getitem__(self, index):
        """Get dataset sample.

        Args:
            index (int): sample index

        Returns:
            tuple: points, side information, fundamental matrix, virtual points 1, virtual points 2,
                  ground truth rotation and translation
        """
        pts = self.pts[index]
        R_gt = self.R_gt[index]
        t_gt = self.t_gt[index]
        K = self.intrinsics[index]

        # normalize side information
        side_info = pts[:, 4:]
        
        pts = pts[:, :4]

        return (np.float32(pts[:, :4]), side_info, np.float32(R_gt), np.float32(t_gt), np.float32(K))
    
    def __len__(self):
        """Get length of dataset.

        Returns:
            int: length
        """
        return len(self.R_gt)


