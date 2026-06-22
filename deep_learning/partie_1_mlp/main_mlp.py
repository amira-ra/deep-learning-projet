
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Importer nos modules
from data_prep import prepare_data
from models import MLPSequential, MLPClass, inspect_model_parameters, initialize_weights
from train import get_device, train_model, evaluate_model, load_best_model


def main():
    # Étape 1 : Préparer les données
    print("="*50)
    print("Préparation des données...")
    print("="*50)
    train_loader, val_loader, test_loader, input_dim = prepare_data(batch_size=32)
    device = get_device()
    
    # Étape 2 : Définir les architectures et les stratégies d'initialisation
    architectures = [
        ("MLPSequential", MLPSequential),
        ("MLPClass", MLPClass)
    ]
    init_strategies = ['gaussian', 'constant', 'xavier']
    
    best_config = None
    best_val_loss = float('inf')
    results = []  # Pour stocker les résultats de chaque configuration
    
    # Étape 3 : Entraîner toutes les configurations
    print("\n" + "="*50)
    print("Entraînement des configurations...")
    print("="*50)
    
    for arch_name, arch_class in architectures:
        for strategy in init_strategies:
            print(f"\n--- Configuration : {arch_name} + Initialisation {strategy} ---")
            
            # Initialiser le modèle
            model = arch_class(input_dim)
            initialize_weights(model, strategy)
            
            # Entraîner le modèle
            save_path = f"best_{arch_name}_{strategy}.pth"
            _, _, val_losses = train_model(
                model, train_loader, val_loader, device,
                epochs=50, lr=0.001, save_path=save_path
            )
            
            # Stocker la meilleure perte de validation pour cette configuration
            final_best_val_loss = min(val_losses)
            results.append({
                "architecture": arch_name,
                "initialization": strategy,
                "best_val_loss": final_best_val_loss
            })
            
            # Mettre à jour la meilleure configuration globale
            if final_best_val_loss < best_val_loss:
                best_val_loss = final_best_val_loss
                best_config = {
                    "architecture": arch_name,
                    "initialization": strategy,
                    "model_class": arch_class,
                    "save_path": save_path
                }
    
    # Afficher les résultats de toutes les configurations
    print("\n" + "="*50)
    print("Résultats des configurations :")
    print("="*50)
    for res in results:
        print(f"{res['architecture']} + {res['initialization']}: Perte Val = {res['best_val_loss']:.4f}")
    
    print(f"\nMeilleure configuration : {best_config['architecture']} + {best_config['initialization']} (Perte Val = {best_val_loss:.4f})")
    
    # Étape 4 : Évaluer la meilleure configuration sur le test set
    print("\n" + "="*50)
    print("Évaluation sur le Test Set...")
    print("="*50)
    
    # Charger le meilleur modèle
    best_model = load_best_model(
        best_config["model_class"],
        input_dim,
        save_path=best_config["save_path"]
    )
    
    # Évaluer
    predictions, labels, test_loss = evaluate_model(best_model, test_loader, device)
    
    # Calculer les métriques
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    cm = confusion_matrix(labels, predictions)
    
    # Afficher les métriques
    print(f"\nPerte Test : {test_loss:.4f}")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")
    
    # Afficher la matrice de confusion
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Bénin', 'Malin'], 
                yticklabels=['Bénin', 'Malin'])
    plt.xlabel('Prédictions')
    plt.ylabel('Vrais labels')
    plt.title('Matrice de Confusion')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("\nMatrice de confusion sauvegardée dans 'confusion_matrix.png'")
    plt.show()


if __name__ == "__main__":
    main()
