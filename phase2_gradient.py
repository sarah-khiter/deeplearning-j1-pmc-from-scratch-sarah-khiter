import numpy as np
import matplotlib
matplotlib.use('Agg') # Pour générer l'image sans interface graphique
import matplotlib.pyplot as plt

# Données
X = np.array([[0.2, 0.1], [0.8, 0.9], [0.3, 0.7], [0.9, 0.2]])
y = np.array([0, 1, 1, 0])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def compute_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Initialisation
np.random.seed(42)
w = np.random.randn(2) * 0.01 
b = 0.0
learning_rate = 0.1
n_epochs = 50
losses = []

print("Début de l'entraînement...")

for epoch in range(n_epochs):
    # --- FORWARD PASS ---
    z = np.dot(X, w) + b
    y_pred = sigmoid(z)
    
    # Calcul et stockage de la loss
    loss = compute_loss(y, y_pred)
    losses.append(loss)
    
    # --- BACKPROPAGATION (Gradient Descent) ---
    # L'erreur résiduelle
    error = y_pred - y
    
    # Calcul des dérivées partielles (Gradients)
    n = len(y)
    dw = np.dot(X.T, error) / n
    db = np.mean(error)
    
    # --- MISE À JOUR DES PARAMÈTRES ---
    w -= learning_rate * dw
    b -= learning_rate * db
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | w: {w.round(3)} | b: {b:.3f}")

# Visualisation
plt.figure(figsize=(8, 4))
plt.plot(losses, label='Loss BCE')
plt.xlabel("Epoch")
plt.ylabel("Loss BCE")
plt.title(f"Convergence (LR={learning_rate})")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.savefig("phase2_loss_curve.png", dpi=100, bbox_inches='tight')

print(f"\nCourbe sauvegardée : phase2_loss_curve.png")
print(f"Loss finale : {losses[-1]:.4f}")