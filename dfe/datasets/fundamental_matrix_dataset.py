"""Base class for a fundamental matrix estimation dataset.
"""
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from torch.utils.data import Dataset
import numpy as np
import cv2


class FundamentalMatrixDataset(Dataset):
    """Base class for a fundamental matrix estimation dataset.
    """

    def __init__(self, num_points, step=0.01):
        """Init.

        Args:
            num_points (int): number of points per sample
            step (float, optional): Step size of virtual evaluation points. Defaults to 0.01.
        """
        self.num_points = num_points
        self.step = step

        self.pts = []
        self.F = []

        self.size_1 = []
        self.size_2 = []

        self.num_points_eval = 0
        self.pts1_virt = []
        self.pts2_virt = []
        self.pts1_grid = []
        self.pts2_grid = []


    def __getitem__(self, index):
        """Get dataset sample.

        Args:
            index (int): sample index

        Returns:
            tuple: points, side information, fundamental matrix, virtual points 1, virtual points 2
        """
        pts = self.pts[index]
        F = self.F[index]

        # print(self.img_paths[index])

        # add data if too small for training
        if self.num_points > 0 and pts.shape[0] < self.num_points:
            while pts.shape[0] < self.num_points:
                num_missing = self.num_points - pts.shape[0]
                idx = np.random.permutation(pts.shape[0])[:num_missing]

                pts_pert = pts[idx]
                pts = np.concatenate((pts, pts_pert), 0)

        # normalize side information
        side_info = pts[:, 4:] / np.amax(pts[:, 4:], 0)

        pts = pts[:, :4]

        # remove data if too big for training
        if self.num_points > 0 and (pts.shape[0] > self.num_points):
            idx = np.random.permutation(pts.shape[0])[: self.num_points]

            pts = pts[idx, :]
            side_info = side_info[idx]

        return (np.float32(pts[:, :4]), side_info, np.float32(F))

    def __len__(self):
        """Get length of dataset.

        Returns:
            int: length
        """
        return len(self.F)
