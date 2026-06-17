
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

# Importer les modules créés
from models import LeNet, LeNetImproved, SimpleMLP, get_feature_maps, count_parameters
from train import get_device, train_model, evaluate, load_best_model, plot_training_curves
from manual_ops import corr2d, manual_max_pool2d, manual_avg_pool2d, compare_with_pytorch


class AblationCNN(nn.Module):
    """
    Petit CNN configurable pour l'étude d'ablation (Section C).
    """
    def __init__(self, padding=2, stride=1, pool_type='max', num_filters1=6, num_filters2=16):
        super(AblationCNN, self).__init__()
        self.padding = padding
        self.stride = stride
        self.pool_type = pool_type
        
        # Couche 1
        self.conv1 = nn.Conv2d(1, num_filters1, kernel_size=5, padding=padding, stride=stride)
        self.relu1 = nn.ReLU()
        if pool_type == 'max':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Couche 2 (réduite pour l'ablation)
        self.conv2 = nn.Conv2d(num_filters1, num_filters2, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected
        self.flatten = nn.Flatten()
        # Calcul de la dimension d'entrée (pour 28x28 FashionMNIST)
        with torch.no_grad():
            dummy = torch.randn(1, 1, 28, 28)
            dummy = self.pool1(self.relu1(self.conv1(dummy)))
            dummy = self.pool2(self.relu2(self.conv2(dummy)))
            self.fc_input_dim = dummy.numel()
        
        self.fc1 = nn.Linear(self.fc_input_dim, 84)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(84, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x


def run_ablation(config_name, train_loader, val_loader, device, padding=2, stride=1, pool_type='max', num_filters1=6, num_filters2=16, epochs=10):
    """
    Fonction pour exécuter une configuration d'ablation (Section C).
    """
    print(f"\n--- Ablation: {config_name} ---")
    model = AblationCNN(padding=padding, stride=stride, pool_type=pool_type, num_filters1=num_filters1, num_filters2=num_filters2).to(device)
    num_params = count_parameters(model)
    
    start_time = time.time()
    history = train_model(
        model, train_loader, val_loader, device,
        epochs=epochs, lr=0.001,
        save_path=f"ablation_{config_name}.pth"
    )
    train_time = time.time() - start_time
    
    # Évaluer sur la validation
    final_val_acc = history["val_accs"][-1]
    return {
        "config": config_name,
        "num_params": num_params,
        "val_acc": final_val_acc,
        "train_time": train_time
    }


def main():
    print("=" * 80)
    print("Partie II: CNN pour classification d'images (FashionMNIST)")
    print("=" * 80)

    # Préparer les données (FashionMNIST)
    print("\n--- Préparation des données ---")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalisation FashionMNIST standard
    ])

    # Charger le dataset complet
    full_train_dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=False, download=True, transform=transform
    )

    # Séparer en train (80%) et val (20%)
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    # Créer les DataLoaders
    batch_size = 64
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"Train: {len(train_dataset)} images")
    print(f"Val:   {len(val_dataset)} images")
    print(f"Test:  {len(test_dataset)} images")

    # Initialiser le périphérique
    device = get_device()
    print(f"\nUtilisation de: {device}")

    # === Section A: Démonstration des opérations manuelles ===
    print("\n" + "=" * 80)
    print("Section A: Démonstration des opérations manuelles")
    print("=" * 80)
    
    # a. corr2d sur 4x4 avec noyau 2x2
    print("\na. Corrélation croisée 2D")
    X = torch.tensor([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, 16.0]
    ])
    K = torch.tensor([
        [0.0, 1.0],
        [2.0, 3.0]
    ])
    print(f"Input X:\n{X.numpy()}")
    print(f"\nNoyau K:\n{K.numpy()}")
    Y_manual = corr2d(X, K)
    print(f"\nRésultat corr2d manuel:\n{Y_manual.numpy()}")
    
    # Calcul taille de sortie
    H_out = X.shape[0] - K.shape[0] + 1
    W_out = X.shape[1] - K.shape[1] + 1
    print(f"\nTaille de sortie (formule): H_out = {H_out}, W_out = {W_out}")
    print(f"Taille de sortie (réelle): {Y_manual.shape}")
    
    # b. Comparaison pooling manuel vs PyTorch
    print("\nb. Comparaison pooling")
    X_pool = torch.tensor([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, 16.0]
    ])
    print(f"Input pooling:\n{X_pool.numpy()}")
    
    print("\nMax Pooling (2x2):")
    Y_max_manual = manual_max_pool2d(X_pool, (2,2))
    print(f"Manuel:\n{Y_max_manual.numpy()}")
    max_pool = nn.MaxPool2d(2,2)
    Y_max_torch = max_pool(X_pool.unsqueeze(0).unsqueeze(0)).squeeze()
    print(f"PyTorch:\n{Y_max_torch.numpy()}")
    print(f"Égalité: {torch.allclose(Y_max_manual, Y_max_torch)}")
    
    print("\nAvg Pooling (2x2):")
    Y_avg_manual = manual_avg_pool2d(X_pool, (2,2))
    print(f"Manuel:\n{Y_avg_manual.numpy()}")
    avg_pool = nn.AvgPool2d(2,2)
    Y_avg_torch = avg_pool(X_pool.unsqueeze(0).unsqueeze(0)).squeeze()
    print(f"PyTorch:\n{Y_avg_torch.numpy()}")
    print(f"Égalité: {torch.allclose(Y_avg_manual, Y_avg_torch)}")
    
    # Test complet avec compare_with_pytorch
    print("\nc. Test complet avec compare_with_pytorch")
    dummy_batch = torch.randn(2,1,6,6)
    compare_with_pytorch(dummy_batch)

    # === Section B: Étude expérimentale architecturale ===
    print("\n" + "=" * 80)
    print("Section B: Étude expérimentale architecturale")
    print("=" * 80)
    
    models_to_train = [
        ("SimpleMLP", SimpleMLP(input_dim=28*28, num_classes=10)),
        ("LeNet", LeNet(num_classes=10)),
        ("LeNetImproved", LeNetImproved(num_classes=10))
    ]
    
    architecture_results = []
    best_model = None
    best_model_name = ""
    
    for name, model in models_to_train:
        print(f"\n--- Entraînement de {name} ---")
        model = model.to(device)
        num_params = count_parameters(model)
        
        start_time = time.time()
        history = train_model(
            model, train_loader, val_loader, device,
            epochs=15, lr=0.001,
            save_path=f"best_{name}.pth"
        )
        train_time = time.time() - start_time
        
        # Tracer les courbes
        plot_training_curves(history, title=name, save_path=f"curves_{name}.png")
        
        # Charger le meilleur modèle et évaluer sur le test set
        model = load_best_model(model, f"best_{name}.pth").to(device)
        test_loss, test_acc = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
        
        # Enregistrer les résultats
        architecture_results.append({
            "Modèle": name,
            "Test Accuracy (%)": f"{test_acc:.2f}",
            "Nb Paramètres": num_params,
            "Temps d'entraînement (s)": f"{train_time:.2f}"
        })
        
        # Mettre à jour le meilleur modèle
        if best_model is None or test_acc > float(best_model_name.split()[0]) if best_model_name else True:
            best_model = model
            best_model_name = f"{test_acc:.2f}% - {name}"
    
    # Afficher le tableau comparatif
    print("\nTableau comparatif des architectures:")
    print(tabulate(architecture_results, headers="keys", tablefmt="grid"))
    
    # Sauvegarder le tableau en image (simple)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')
    table_data = [[d["Modèle"], d["Test Accuracy (%)"], d["Nb Paramètres"], d["Temps d'entraînement (s)"]] for d in architecture_results]
    ax.table(cellText=table_data, colLabels=list(architecture_results[0].keys()), loc='center', cellLoc='center')
    plt.savefig("table_architectures.png", dpi=300, bbox_inches='tight')
    plt.close()

    # === Section C: Étude de l'influence des hyperparamètres ===
    print("\n" + "=" * 80)
    print("Section C: Étude de l'influence des hyperparamètres")
    print("=" * 80)
    
    ablation_configs = [
        {"config_name": "Baseline", "padding": 2, "stride": 1, "pool_type": "max", "num_filters1": 6, "num_filters2": 16},
        {"config_name": "No Padding", "padding": 0, "stride": 1, "pool_type": "max", "num_filters1": 6, "num_filters2": 16},
        {"config_name": "Stride 2", "padding": 2, "stride": 2, "pool_type": "max", "num_filters1": 6, "num_filters2": 16},
        {"config_name": "Avg Pool", "padding": 2, "stride": 1, "pool_type": "avg", "num_filters1": 6, "num_filters2": 16},
        {"config_name": "More Filters", "padding": 2, "stride": 1, "pool_type": "max", "num_filters1": 12, "num_filters2": 32}
    ]
    
    ablation_results = []
    for config in ablation_configs:
        result = run_ablation(**config, train_loader=train_loader, val_loader=val_loader, device=device)
        ablation_results.append({
            "Configuration": result["config"],
            "Val Accuracy (%)": f"{result['val_acc']:.2f}",
            "Nb Paramètres": result["num_params"],
            "Temps (s)": f"{result['train_time']:.2f}"
        })
    
    # Afficher le tableau ablation
    print("\nTableau de l'étude d'ablation:")
    print(tabulate(ablation_results, headers="keys", tablefmt="grid"))
    
    # Sauvegarder le tableau
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')
    table_data = [[d["Configuration"], d["Val Accuracy (%)"], d["Nb Paramètres"], d["Temps (s)"]] for d in ablation_results]
    ax.table(cellText=table_data, colLabels=list(ablation_results[0].keys()), loc='center', cellLoc='center')
    plt.savefig("table_ablation.png", dpi=300, bbox_inches='tight')
    plt.close()

    # === Section D: Visualisation des feature maps ===
    print("\n" + "=" * 80)
    print("Section D: Visualisation des feature maps")
    print("=" * 80)
    
    # Prendre une image du test set
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    img = images[0].unsqueeze(0).to(device)
    true_label = class_names[labels[0]]
    
    # Afficher l'image d'origine
    plt.figure(figsize=(4,4))
    plt.imshow(img.squeeze().cpu().numpy(), cmap='gray')
    plt.title(f"Image d'origine: {true_label}")
    plt.axis('off')
    plt.savefig("original_image.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Charger LeNet et obtenir les feature maps
    lenet = LeNet(num_classes=10).to(device)
    lenet = load_best_model(lenet, "best_LeNet.pth")
    lenet.eval()
    with torch.no_grad():
        fmaps = get_feature_maps(lenet, img)
    
    # Afficher les feature maps de la première couche conv
    first_conv_fmaps = fmaps[0].squeeze().cpu().numpy()  # (6, 28, 28) pour LeNet
    num_fmaps = first_conv_fmaps.shape[0]
    fig, axes = plt.subplots(1, num_fmaps, figsize=(num_fmaps * 2, 2))
    for i in range(num_fmaps):
        ax = axes[i]
        ax.imshow(first_conv_fmaps[i], cmap='viridis')
        ax.set_title(f"Filtre {i+1}")
        ax.axis('off')
    plt.suptitle("Feature Maps - Première couche Conv (LeNet)", y=1.02)
    plt.tight_layout()
    plt.savefig("feature_maps.png", dpi=300, bbox_inches='tight')
    plt.show()

    # === Résumé final ===
    print("\n" + "=" * 80)
    print("RÉSUMÉ FINAL")
    print("=" * 80)
    print("Section A: Opérations manuelles vérifiées")
    print("Section B: Meilleur modèle architectural:", best_model_name)
    print("Section C: Étude d'ablation terminée")
    print("Section D: Feature maps visualisées et sauvegardées")
    print("\nToutes les figures ont été sauvegardées en PNG!")


if __name__ == "__main__":
    main()
