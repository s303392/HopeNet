import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

# Carica i dati dal file .npy
predictions = np.load('./predictions/hopenet_predictions.npy')
labels = np.load('./predictions/hopenet_labels.npy')  # Carica il file delle etichette

# Visualizza la forma degli array per capire la struttura dei dati
print("Predictions shape:", predictions.shape)
print("Labels shape:", labels.shape)

# Stampa un esempio di predizione e di etichetta
print("Example prediction:", predictions[0])
print("Example label:", labels[0])

# Seleziona un esempio di predizione e di etichetta
pred_points = predictions[0]
label_points = labels[0]

# Crea una figura 3D interattiva
fig = go.Figure()

# Aggiungi le predizioni al grafico
fig.add_trace(go.Scatter3d(
    x=pred_points[:, 0],
    y=pred_points[:, 1],
    z=pred_points[:, 2],
    mode='markers',
    marker=dict(color='red'),
    name='Predictions'
))

# Aggiungi le etichette al grafico
fig.add_trace(go.Scatter3d(
    x=label_points[:, 0],
    y=label_points[:, 1],
    z=label_points[:, 2],
    mode='markers',
    marker=dict(color='blue'),
    name='Ground Truth'
))

# Visualizza la figura
fig.show()
