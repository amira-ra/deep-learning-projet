
import torch
import torch.nn as nn
import torch.optim as optim


def get_device():
    """
    Récupère le périphérique disponible (GPU si disponible, sinon CPU).
    
    Returns:
        torch.device: Périphérique à utiliser
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation du périphérique : {device}")
    return device


def train_model(model, train_loader, val_loader, device, epochs=50, lr=0.001, save_path="best_model.pth"):
    """
    Entraîne un modèle et sauvegarde le meilleur état basé sur la perte de validation.
    
    Args:
        model (nn.Module): Modèle à entraîner
        train_loader (DataLoader): DataLoader pour l'entraînement
        val_loader (DataLoader): DataLoader pour la validation
        device (torch.device): Périphérique à utiliser
        epochs (int): Nombre d'époques
        lr (float): Taux d'apprentissage
        save_path (str): Chemin pour sauvegarder le meilleur modèle
        
    Returns:
        tuple: (model, train_losses, val_losses)
    """
    # Déplacer le modèle sur le périphérique
    model = model.to(device)
    
    # Définir la perte et l'optimiseur
    #BCELoss = la fonction qui mesure l'erreur (pour classification binaire : bénin/malin).
#Adam = l'algorithme qui corrige les poids. lr=0.001 = la vitesse de correction.

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Meilleure perte de validation (initialisée à l'infini
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Phase d'entraînement 50 fois
        model.train()
        train_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Remettre à zéro les gradients
            optimizer.zero_grad()
            
            # Passage en avant, calcul de la perte, passage en arrière, optimisation
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
        
        # Calcul de la perte moyenne d'entraînement
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Phase de validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
        
        # Calcul de la perte moyenne de validation
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        
        # Affichage des résultats de l'époque
        print(f"Époque {epoch+1}/{epochs} - Perte Train : {train_loss:.4f} - Perte Val : {val_loss:.4f}")
        
        # Sauvegarder le meilleur modèle
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  -> Meilleur modèle sauvegardé (perte Val : {val_loss:.4f})")
    
    return model, train_losses, val_losses


def evaluate_model(model, test_loader, device):
    """
    Évalue un modèle sur un dataset de test.
    
    Args:
        model (nn.Module): Modèle à évaluer
        test_loader (DataLoader): DataLoader pour le test
        device (torch.device): Périphérique à utiliser
        
    Returns:
        tuple: (predictions, labels, test_loss)
    """
    model = model.to(device)
    model.eval()
    criterion = nn.BCELoss()
    test_loss = 0.0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            
            # Convertir les probabilités en classes (seuil 0.5) Si ≥ 0.5 → malin. Si < 0.5 → bénin.
            predictions = (outputs >= 0.5).float()
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_loss /= len(test_loader.dataset)
    
    return all_predictions, all_labels, test_loss


def load_best_model(model_class, input_dim, save_path="best_model.pth"):
    """
    Recharge le meilleur modèle sauvegardé.
    
    Args:
        model_class (nn.Module): Classe du modèle à initialiser
        input_dim (int): Dimension des entrées
        save_path (str): Chemin du modèle sauvegardé
        
    Returns:
        nn.Module: Modèle chargé avec les meilleurs poids
    """
    model = model_class(input_dim)
    model.load_state_dict(torch.load(save_path))
    return model


if __name__ == "__main__":
    # Test rapide des fonctions
    print("Module train.py chargé avec succès!")
