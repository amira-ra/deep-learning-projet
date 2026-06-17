
import time
import torch
import torch.nn as nn
import torch.optim as optim


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_language_model(model, dataloader, criterion, optimizer, device, clip=1.0):
    """Entraîne un modèle de langage pour une époque avec Gradient Clipping."""
    model.train()
    epoch_loss = 0
    
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        # Initialiser l'état caché
        hidden = model.init_hidden(inputs.shape[0], device)
        
        # Forward pass
        logits, hidden = model(inputs, hidden)
        
        # Calculer la perte
        # logits: (batch_size, seq_len, vocab_size)
        # targets: (batch_size, seq_len)
        loss = criterion(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        
        # Backward pass
        loss.backward()
        
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(dataloader)


def evaluate_language_model(model, dataloader, criterion, device):
    """Évalue un modèle de langage."""
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            hidden = model.init_hidden(inputs.shape[0], device)
            logits, hidden = model(inputs, hidden)
            
            loss = criterion(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
            epoch_loss += loss.item()
            
    return epoch_loss / len(dataloader)


def train_seq2seq(model, dataloader, criterion, optimizer, device, clip=1.0):
    """Entraîne un modèle Seq2Seq pour une époque avec Gradient Clipping."""
    model.train()
    epoch_loss = 0
    
    for src, trg in dataloader:
        src = src.to(device)
        trg = trg.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(src, trg)
        
        # Calculer la perte (ignorer le premier token <SOS>)
        # outputs: (batch_size, trg_len, vocab_size)
        # trg: (batch_size, trg_len)
        output_dim = outputs.shape[-1]
        outputs = outputs[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)
        
        loss = criterion(outputs, trg)
        
        # Backward pass
        loss.backward()
        
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(dataloader)


def evaluate_seq2seq(model, dataloader, criterion, device):
    """Évalue un modèle Seq2Seq."""
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for src, trg in dataloader:
            src = src.to(device)
            trg = trg.to(device)
            
            outputs = model(src, trg, teacher_forcing_ratio=0.0)
            
            output_dim = outputs.shape[-1]
            outputs = outputs[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)
            
            loss = criterion(outputs, trg)
            epoch_loss += loss.item()
            
    return epoch_loss / len(dataloader)


def train_language_model_epochs(model, train_dataloader, val_dataloader, vocab_size, device,
                                 epochs=20, lr=0.001, clip=1.0):
    """Entraîne un modèle de langage pour plusieurs époques."""
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 0 = PAD
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses = []
    val_perplexities = []
    epoch_times = []
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        start_time = time.time()
        
        train_loss = train_language_model(model, train_dataloader, criterion, optimizer, device, clip)
        val_loss = evaluate_language_model(model, val_dataloader, criterion, device)
        val_perplexity = torch.exp(torch.tensor(val_loss)).item()
        
        end_time = time.time()
        epoch_time = end_time - start_time
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_perplexities.append(val_perplexity)
        epoch_times.append(epoch_time)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
        print(f"Époque {epoch+1}/{epochs}:")
        print(f"  Perte Train: {train_loss:.4f} | Perte Val: {val_loss:.4f}")
        print(f"  Perplexité Val: {val_perplexity:.2f} | Temps: {epoch_time:.2f}s")
        
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_perplexities': val_perplexities,
        'epoch_times': epoch_times,
        'final_val_loss': val_losses[-1],
        'final_val_perplexity': val_perplexities[-1],
        'avg_epoch_time': sum(epoch_times) / len(epoch_times),
        'best_val_loss': best_val_loss
    }


if __name__ == "__main__":
    print("Module train.py prêt!")
