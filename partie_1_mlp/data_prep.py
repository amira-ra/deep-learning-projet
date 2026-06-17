
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader


def prepare_data(batch_size=32):
    """
    Prépare les données pour l'entraînement d'un MLP sur le dataset Breast Cancer Wisconsin.
    
    Étapes :
    1. Chargement du dataset
    2. Nettoyage (vérification des valeurs manquantes)
    3. Normalisation (StandardScaler)
    4. Séparation Train/Validation/Test (70/15/15)
    5. Création des DataLoaders PyTorch
    
    Args:
        batch_size (int): Taille des lots pour les DataLoaders
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, input_dim)
    """
    # Étape 1 : Charger le dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    target_names = data.target_names
    
    # Étape 2 : Nettoyage - vérifier les valeurs manquantes
    df = pd.DataFrame(X, columns=feature_names)
    print(f"Valeurs manquantes dans le dataset : {df.isnull().sum().sum()}")
    
    # Étape 3 : Normalisation StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Étape 4 : Séparation Train/Validation/Test (70/15/15)
    # D'abord séparer Train (70%) et le reste (30%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )
    # Puis séparer le reste en Validation (15%) et Test (15%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"Taille Train : {len(X_train)}")
    print(f"Taille Validation : {len(X_val)}")
    print(f"Taille Test : {len(X_test)}")
    
    # Convertir en tenseurs PyTorch
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    
    # Créer des datasets et dataloaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    input_dim = X_train.shape[1]
    
    return train_loader, val_loader, test_loader, input_dim


if __name__ == "__main__":
    # Test de la fonction
    train_loader, val_loader, test_loader, input_dim = prepare_data()
    print(f"Dimension des entrées : {input_dim}")
