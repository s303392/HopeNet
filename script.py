import numpy as np
import os

def check_npy_files(root, load_set):
    try:
        images = np.load(os.path.join(root, f'images-{load_set}.npy'))
        points2d = np.load(os.path.join(root, f'points2d-{load_set}.npy'))
        points3d = np.load(os.path.join(root, f'points3d-{load_set}.npy'))
        
        print(f'Number of images: {len(images)}')
        print(f'Number of 2D points: {len(points2d)}')
        print(f'Number of 3D points: {len(points3d)}')
    except Exception as e:
        print(f'Error: {e}')

# Esempio di utilizzo
check_npy_files('./datasets/fhad', 'test')  
