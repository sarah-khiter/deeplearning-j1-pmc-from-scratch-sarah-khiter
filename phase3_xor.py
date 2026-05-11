import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 1. Données XOR (Incontournables)
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y_xor = np.array([0, 1, 1, 0])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def compute_loss_bce(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# 2. Initialisation 2-2-1 (C'est ici qu'on "fixe" le problème)
# On utilise une seed différente et une distribution uniforme plus large
np.random.seed(1) 
W1 = np.random.uniform(low=-2.0, high=2.0, size=(2, 2))
b1 = np.zeros(2)
W2 = np.random.uniform(low=-2.0, high=2.0, size=(2, 1))
b2 = np.zeros(1)

# Paramètres musclés pour sortir du plateau
learning_rate = 1.0  
n_epochs = 10001
losses = []

print("Entraînement du MLP XOR : Objectif 100%...")

for epoch in range(n_epochs):
    # --- FORWARD PASS ---
    z1 = np.dot(X_xor, W1) + b1
    a1 = sigmoid(z1)
    
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)
    y_pred = a2.flatten()
    
    # Loss
    loss = compute_loss_bce(y_xor, y_pred)
    losses.append(loss)
    
    # --- BACKPROPAGATION ---
    n = len(y_xor)
    error2 = a2 - y_xor.reshape(-1, 1)
    
    # Gradients Sortie
    dW2 = np.dot(a1.T, error2) / n
    db2 = np.mean(error2, axis=0)
    
    # Gradients Couche Cachée (La clé du XOR)
    error1 = np.dot(error2, W2.T) * (a1 * (1 - a1))
    dW1 = np.dot(X_xor.T, error1) / n
    db1 = np.mean(error1, axis=0)
    
    # --- UPDATE ---
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    
    # Affichage régulier
    if epoch % 2000 == 0:
        acc = np.mean((y_pred > 0.5) == y_xor)
        print(f"Epoch {epoch:5d} | Loss: {loss:.4f} | Accuracy: {acc:.2%}")

# --- ANALYSE FINALE ---
final_acc = np.mean((y_pred > 0.5) == y_xor)
print(f"\n--- RÉSULTAT FINAL ---")
print(f"Loss finale : {losses[-1]:.4f}")
print(f"Accuracy finale : {final_acc:.2%}")

if final_acc == 1.0:
    print("✅ Succès : Le XOR est résolu !")
else:
    print("❌ Échec : Toujours bloqué. Essayez de changer la seed.")

# --- PLOT DE LA FRONTIÈRE ---
xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
grid = np.c_[xx.ravel(), yy.ravel()]
z1g = sigmoid(np.dot(grid, W1) + b1)
z2g = sigmoid(np.dot(z1g, W2) + b2).reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, z2g, alpha=0.4, cmap='RdBu')
plt.scatter(X_xor[:, 0], X_xor[:, 1], c=y_xor, s=100, edgecolors='k', cmap='RdBu')
plt.title(f"XOR Résolu - Accuracy: {final_acc:.2%}")
plt.savefig("phase3_xor_success.png")