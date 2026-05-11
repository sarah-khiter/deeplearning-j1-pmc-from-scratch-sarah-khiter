import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def generate_spiral(n_points=400, noise=0.15, seed=42):
    np.random.seed(seed)
    n = n_points // 2
    theta = np.sqrt(np.random.rand(n)) * 4 * np.pi 
    
    # Spirale 1
    X0 = np.c_[-np.cos(theta) * theta + np.random.randn(n) * noise,
               np.sin(theta) * theta + np.random.randn(n) * noise]
    # Spirale 2
    X1 = np.c_[np.cos(theta) * theta + np.random.randn(n) * noise,
               -np.sin(theta) * theta + np.random.randn(n) * noise]
    
    X = np.vstack([X0, X1]) / 15.0 # Normalisation pour aider la convergence
    y = np.hstack([np.zeros(n), np.ones(n)])
    return X, y

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(float)

def bce_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# 1. Préparation des données
X, y = generate_spiral(n_points=400, noise=0.15)
n_samples = len(y)

# 2. Initialisation He (Améliorée)
np.random.seed(42)
# On utilise 64 neurones par couche
W1 = np.random.randn(2, 64) * np.sqrt(2 / 2)
b1 = np.zeros(64)
W2 = np.random.randn(64, 64) * np.sqrt(2 / 64)
b2 = np.zeros(64)
W3 = np.random.randn(64, 1) * np.sqrt(2 / 64)
b3 = np.zeros(1)

# 3. Hyperparamètres ajustés
lr = 0.05 # Augmenté pour accélérer la convergence
n_epochs = 5000 # Augmenté pour laisser le temps de sculpter la spirale
losses = []

print("Entraînement Profond (2-64-64-1) - Objectif > 90%...")

for epoch in range(n_epochs):
    # --- FORWARD ---
    z1 = np.dot(X, W1) + b1
    a1 = relu(z1)
    
    z2 = np.dot(a1, W2) + b2
    a2 = relu(z2)
    
    z3 = np.dot(a2, W3) + b3
    y_pred_raw = sigmoid(z3)
    y_pred = y_pred_raw.flatten()
    
    # Loss
    loss = bce_loss(y, y_pred)
    losses.append(loss)
    
    # --- BACKWARD ---
    err3 = y_pred_raw - y.reshape(-1, 1)
    dW3 = np.dot(a2.T, err3) / n_samples
    db3 = np.mean(err3, axis=0)
    
    err2 = np.dot(err3, W3.T) * relu_grad(z2)
    dW2 = np.dot(a1.T, err2) / n_samples
    db2 = np.mean(err2, axis=0)
    
    err1 = np.dot(err2, W2.T) * relu_grad(z1)
    dW1 = np.dot(X.T, err1) / n_samples
    db1 = np.mean(err1, axis=0)
    
    # --- UPDATE ---
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2
    W3 -= lr * dW3
    b3 -= lr * db3
    
    if epoch % 1000 == 0:
        acc = np.mean((y_pred > 0.5) == y)
        print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | Accuracy: {acc:.2%}")

# --- RÉSULTATS ---
final_acc = np.mean((y_pred > 0.5) == y)
print(f"\nLoss finale : {losses[-1]:.4f}")
print(f"Accuracy finale : {final_acc:.2%}")

# --- PLOT FINAL ---
h = 0.02
xx, yy = np.meshgrid(np.arange(X[:, 0].min() - 0.1, X[:, 0].max() + 0.1, h),
                     np.arange(X[:, 1].min() - 0.1, X[:, 1].max() + 0.1, h))
grid = np.c_[xx.ravel(), yy.ravel()]
a1g = relu(np.dot(grid, W1) + b1)
a2g = relu(np.dot(a1g, W2) + b2)
zg = sigmoid(np.dot(a2g, W3) + b3).reshape(xx.shape)

plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, zg, alpha=0.5, cmap='RdBu')
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdBu', s=20, edgecolors='k')
plt.title(f"Spirale - Accuracy: {final_acc:.2%}")
plt.savefig("phase4_spirale_success.png")