
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


def get_device():
    """
    Retourne le périphérique disponible (GPU si disponible, sinon CPU).
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(model, loader, criterion, optimizer, device):
    """
    Entraîne le modèle pour une époque.
    Args:
        model: modèle à entraîner
        loader: DataLoader d'entraînement
        criterion: fonction de perte
        optimizer: optimiseur
        device: périphérique (cuda/cpu)
    Returns:
        (train_loss, train_accuracy): perte moyenne et accuracy moyenne sur l'époque
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Remettre à zéro les gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass et optimisation
        loss.backward()
        optimizer.step()
        
        # Statistiques
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = total_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    """
    Évalue le modèle sur un jeu de données (validation ou test).
    Args:
        model: modèle à évaluer
        loader: DataLoader d'évaluation
        criterion: fonction de perte
        device: périphérique (cuda/cpu)
    Returns:
        (loss, accuracy): perte moyenne et accuracy moyenne
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    avg_loss = total_loss / total
    avg_acc = 100. * correct / total
    return avg_loss, avg_acc


def train_model(model, train_loader, val_loader, device, epochs=20, lr=0.001, save_path="best_cnn.pth"):
    """
    Boucle d'entraînement complète avec sauvegarde du meilleur modèle.
    Args:
        model: modèle à entraîner
        train_loader: DataLoader d'entraînement
        val_loader: DataLoader de validation
        device: périphérique (cuda/cpu)
        epochs: nombre d'époques
        lr: taux d'apprentissage
        save_path: chemin pour sauvegarder le meilleur modèle
    Returns:
        history: dictionnaire avec train_losses, val_losses, train_accs, val_accs
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Meilleur perte de validation initialisée à l'infini
    best_val_loss = float('inf')
    
    # Historique pour les courbes
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(epochs):
        print(f"\n=== Époque {epoch + 1}/{epochs} ===")
        
        # Entraînement
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Évaluation sur la validation
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        # Enregistrer les métriques
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Afficher les résultats
        print(f"Train: Loss = {train_loss:.4f} | Accuracy = {train_acc:.2f}%")
        print(f"Val:   Loss = {val_loss:.4f} | Accuracy = {val_acc:.2f}%")
        
        # Sauvegarder le meilleur modèle
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"→ Meilleur modèle sauvegardé (val_loss = {best_val_loss:.4f})")
    
    # Retourner l'historique
    history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs
    }
    return history


def load_best_model(model, save_path):
    """
    Recharge le state_dict du meilleur modèle sauvegardé.
    """
    model.load_state_dict(torch.load(save_path))
    return model


def plot_training_curves(history, title="", save_path="training_curves.png"):
    """
    Trace les courbes de perte et d'accuracy (train vs validation) avec matplotlib.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Courbe de perte
    ax1.plot(history["train_losses"], label="Train Loss", color="blue")
    ax1.plot(history["val_losses"], label="Val Loss", color="red")
    ax1.set_title(f"Loss - {title}")
    ax1.set_xlabel("Époque")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Courbe d'accuracy
    ax2.plot(history["train_accs"], label="Train Accuracy", color="blue")
    ax2.plot(history["val_accs"], label="Val Accuracy", color="red")
    ax2.set_title(f"Accuracy - {title}")
    ax2.set_xlabel("Époque")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\nCourbes d'entraînement sauvegardées sous: {save_path}")
    plt.show()


if __name__ == "__main__":
    print("=== Test rapide des fonctions d'entraînement ===")
    print("Module prêt à l'emploi!")
