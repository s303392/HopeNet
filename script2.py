import numpy as np

def load_and_print(file_path):
    try:
        data = np.load(file_path, allow_pickle=True)
        print(f'File: {file_path}')
        print(f'Contenuto: {data}')
        print(f'Shape: {data.shape}')
    except Exception as e:
        print(f'Errore nel caricare {file_path}: {e}')

# Verifica i file
load_and_print('./datasets/fhad/images-train.npy')
load_and_print('./datasets/fhad/points2d-train.npy')
load_and_print('./datasets/fhad/points3d-train.npy')

load_and_print('./datasets/fhad/images-val.npy')
load_and_print('./datasets/fhad/points2d-val.npy')
load_and_print('./datasets/fhad/points3d-val.npy')

load_and_print('./datasets/fhad/images-test.npy')
load_and_print('./datasets/fhad/points2d-test.npy')
load_and_print('./datasets/fhad/points3d-test.npy')
