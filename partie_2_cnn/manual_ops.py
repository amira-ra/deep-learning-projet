
import torch
import torch.nn as nn


def corr2d(X, K):
    """
    Corrélation croisée 2D manuelle.
    X: tenseur (H, W) - entrée
    K: tenseur (kH, kW) - noyau/kernel
    Sortie: tenseur (H - kH + 1, W - kW + 1)
    """
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i+h, j:j+w] * K).sum()
    return Y


def corr2d_multi_in(X, K):
    """
    Corrélation croisée 2D avec plusieurs canaux d'entrée.
    X: tenseur (C_in, H, W) - entrée avec C_in canaux
    K: tenseur (C_in, kH, kW) - noyau avec C_in canaux
    Sortie: tenseur (H - kH + 1, W - kW + 1) (somme des corrélations par canal)
    """
    # On calcule la corrélation pour chaque canal et on somme les résultats
    return sum(corr2d(x, k) for x, k in zip(X, K))


def corr2d_multi_in_out(X, K):
    """
    Corrélation croisée 2D avec plusieurs canaux d'entrée ET de sortie.
    X: tenseur (C_in, H, W) - entrée
    K: tenseur (C_out, C_in, kH, kW) - noyaux (un par canal de sortie)
    Sortie: tenseur (C_out, H - kH + 1, W - kW + 1)
    """
    # Pour chaque canal de sortie, on applique corr2d_multi_in et on empile les résultats
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)


def manual_max_pool2d(X, pool_size):
    """
    Max pooling 2D manuel.
    X: tenseur (H, W) - entrée
    pool_size: tuple (pH, pW) - taille de la fenêtre de pooling (stride = pool_size par défaut)
    Sortie: tenseur (H // pH, W // pW)
    """
    pH, pW = pool_size
    H_out = X.shape[0] // pH
    W_out = X.shape[1] // pW
    Y = torch.zeros((H_out, W_out))
    for i in range(H_out):
        for j in range(W_out):
            Y[i, j] = X[i*pH : (i+1)*pH, j*pW : (j+1)*pW].max()
    return Y


def manual_avg_pool2d(X, pool_size):
    """
    Average pooling 2D manuel.
    X: tenseur (H, W) - entrée
    pool_size: tuple (pH, pW) - taille de la fenêtre de pooling (stride = pool_size par défaut)
    Sortie: tenseur (H // pH, W // pW)
    """
    pH, pW = pool_size
    H_out = X.shape[0] // pH
    W_out = X.shape[1] // pW
    Y = torch.zeros((H_out, W_out))
    for i in range(H_out):
        for j in range(W_out):
            Y[i, j] = X[i*pH : (i+1)*pH, j*pW : (j+1)*pW].mean()
    return Y


def compare_with_pytorch(X_batch):
    """
    Compare les résultats manuels avec les couches PyTorch built-in.
    X_batch: tenseur (N, C_in, H, W) - batch d'entrées
    """
    # Paramètres de test
    in_channels = X_batch.shape[1]
    out_channels = 3
    kernel_size = (3, 3)
    pool_size = (2, 2)
    
    # 1. Test Convolution Multi-Entrée/Sortie
    # Initialiser un noyau aléatoire
    K = torch.randn(out_channels, in_channels, *kernel_size)
    
    # Calcul manuel
    manual_conv = torch.stack([
        corr2d_multi_in_out(X_batch[i], K)
        for i in range(X_batch.shape[0])
    ], 0)
    
    # Calcul PyTorch
    conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=False)
    conv.weight.data = K  # Affecter notre noyau manuel à la couche PyTorch
    with torch.no_grad():
        pytorch_conv = conv(X_batch)
    
    # Comparer
    print("=== Test Convolution ===")
    print(f"Forme manuelle: {manual_conv.shape}, Forme PyTorch: {pytorch_conv.shape}")
    print(f"Région allclose (1e-5): {torch.allclose(manual_conv, pytorch_conv, rtol=1e-05, atol=1e-08)}")
    
    # 2. Test Max Pooling
    # Calcul manuel (sur le premier canal de la première image)
    X_single = X_batch[0, 0]
    manual_max = manual_max_pool2d(X_single, pool_size)
    
    # Calcul PyTorch
    max_pool = nn.MaxPool2d(kernel_size=pool_size)
    with torch.no_grad():
        pytorch_max = max_pool(X_single.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
    
    # Comparer
    print("\n=== Test Max Pooling ===")
    print(f"Forme manuelle: {manual_max.shape}, Forme PyTorch: {pytorch_max.shape}")
    print(f"Région allclose (1e-5): {torch.allclose(manual_max, pytorch_max, rtol=1e-05, atol=1e-08)}")
    
    # 3. Test Average Pooling
    manual_avg = manual_avg_pool2d(X_single, pool_size)
    
    avg_pool = nn.AvgPool2d(kernel_size=pool_size)
    with torch.no_grad():
        pytorch_avg = avg_pool(X_single.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
    
    print("\n=== Test Average Pooling ===")
    print(f"Forme manuelle: {manual_avg.shape}, Forme PyTorch: {pytorch_avg.shape}")
    print(f"Région allclose (1e-5): {torch.allclose(manual_avg, pytorch_avg, rtol=1e-05, atol=1e-08)}")


if __name__ == "__main__":
    print("=== Tests des opérations manuelles ===")
    
    # Test corr2d
    print("\n--- Test corr2d ---")
    X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
    K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    print("X:\n", X)
    print("K:\n", K)
    print("Y (corr2d):\n", corr2d(X, K))
    
    # Test corr2d_multi_in
    print("\n--- Test corr2d_multi_in ---")
    X_multi = torch.stack([X, X + 1], 0)
    K_multi = torch.stack([K, K + 1], 0)
    print("Y (multi_in):\n", corr2d_multi_in(X_multi, K_multi))
    
    # Test corr2d_multi_in_out
    print("\n--- Test corr2d_multi_in_out ---")
    K_multi_out = torch.stack([K_multi, K_multi + 2], 0)
    print("Y (multi_in_out):\n", corr2d_multi_in_out(X_multi, K_multi_out))
    
    # Test pooling
    print("\n--- Test pooling ---")
    X_pool = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15.0]])
    print("Max pooling:\n", manual_max_pool2d(X_pool, (2, 2)))
    print("Avg pooling:\n", manual_avg_pool2d(X_pool, (2, 2)))
    
    # Test comparaison PyTorch
    print("\n--- Test comparaison avec PyTorch ---")
    X_batch_test = torch.randn(2, 2, 6, 6)
    compare_with_pytorch(X_batch_test)
