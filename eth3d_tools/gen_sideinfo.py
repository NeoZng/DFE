"""
Script to extract SIFT descriptors, scale, and orientation from COLMAP data
at existing point locations.
"""

import os
import sys
import argparse
import cv2
import numpy as np
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import colmap_read


def extract_sift_features(image_path):
    """Extract SIFT features from an image.
    
    Args:
        image_path (str): Path to the image
        
    Returns:
        tuple: (keypoints, descriptors) or None if image can't be read
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Warning: Could not read image {image_path}")
        return None
    
    # Initialize SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    # Extract keypoints and descriptors
    kps, descs = sift.detectAndCompute(img, None)
    
    if len(kps) > 2500:
        # Sort keypoints by response (strength)
        indices = sorted(range(len(kps)), key=lambda i: kps[i].response, reverse=True)[:2500]
        kps = [kps[i] for i in indices]
        descs = descs[indices]
    
    return kps, descs

def process_colmap_data(colmap_path):
    """Process COLMAP data to extract SIFT information for existing points.
    
    Args:
        colmap_path (str): Path to COLMAP data folder
        dataset_path (str): Path to dataset directory where features will be saved
    """
    # Create output directory (feature folder within dataset path)
    dataset_path = colmap_path
    output_dir = os.path.join(dataset_path, "feature")
    os.makedirs(output_dir, exist_ok=True)
    
    # Read COLMAP data
    cameras = colmap_read.read_cameras_text(os.path.join(colmap_path, "cameras.txt"))
    images = colmap_read.read_images_text(os.path.join(colmap_path, "images.txt"))
    
    # Process each image
    for image_id, image_data in tqdm(images.items(), desc="Processing images"):
        # Extract image path
        image_base_path = os.path.dirname(colmap_path)
        image_path = os.path.join(image_base_path, image_data.name)
        
        # Extract SIFT features directly from the image
        features_result = extract_sift_features(image_path)
        
        if features_result is None:
            print(f"Skipping image {image_data.name} (ID: {image_id}) - could not extract features")
            continue
            
        kps, descs = features_result
        
        if len(kps) == 0:
            print(f"Skipping image {image_data.name} (ID: {image_id}) - no keypoints detected")
            continue
        
        # Save to output file using the original image filename
        image_filename = os.path.basename(image_data.name)
        output_path = os.path.join(output_dir, image_filename.replace('.JPG', '.txt'))
        
        with open(output_path, 'w') as f:
            f.write("# x y descriptor[128] scale orientation\n")
            for i, (kp, desc) in enumerate(zip(kps, descs)):
                x, y = kp.pt
                scale = kp.size
                orientation = kp.angle
                # Format: x y descriptor[128] scale orientation
                descriptor_str = ' '.join([f"{d}" for d in desc])
                f.write(f"{x} {y} {descriptor_str} {scale} {orientation}\n")
        
        print(f"Saved {len(kps)} SIFT features for image {image_data.name} to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate side information from COLMAP data")
    parser.add_argument('--colmap_path', required=True, 
                        help='Path to COLMAP data folder containing cameras.txt, images.txt, etc.')
    args = parser.parse_args()
    
    process_colmap_data(args.colmap_path)


if __name__ == "__main__":
    main()
