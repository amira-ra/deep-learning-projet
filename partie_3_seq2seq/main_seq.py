
import torch
import random
from tabulate import tabulate

from data_prep import (
    Vocabulary, SPECIAL_TOKENS, get_sample_texts,
    prepare_language_modeling_data, pad_sequences
)
from models import LanguageModel, Encoder, Decoder, Seq2Seq, greedy_decode, beam_search_decode
from train import get_device, train_language_model_epochs


def prepare_seq2seq_data(texts, vocab, batch_size=32):
    """Prépare des données pour Seq2Seq (auto-encodeur de phrases)."""
    sequences = []
    for text in texts:
        seq = vocab.text_to_sequence(text, add_eos=True, add_sos=True)
        sequences.append(seq)
    
    def collate_fn(batch):
        padded = pad_sequences(batch, vocab.token2idx['<PAD>'])
        return torch.tensor(padded, dtype=torch.long), torch.tensor(padded, dtype=torch.long)
    
    from torch.utils.data import Dataset, DataLoader
    class SimpleDataset(Dataset):
        def __init__(self, seqs):
            self.seqs = seqs
        def __len__(self):
            return len(self.seqs)
        def __getitem__(self, idx):
            return self.seqs[idx]
    
    dataset = SimpleDataset(sequences)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return dataloader


def main():
    print("="*60)
    print("Partie 3: Modèles de langage et Seq2Seq")
    print("="*60)
    
    # Préparer les données et le vocabulaire
    texts = get_sample_texts()
    vocab = Vocabulary(SPECIAL_TOKENS)
    vocab.build_vocab(texts)
    print(f"\nVocabulaire construit: {len(vocab)} tokens")
    
    # Diviser les données en train/val pour le modèle de langage
    random.shuffle(texts)
    train_texts = texts[:-3]
    val_texts = texts[-3:]
    
    print(f"\nTrain: {len(train_texts)} phrases, Val: {len(val_texts)} phrases")
    
    # Préparer les DataLoaders
    train_dataloader = prepare_language_modeling_data(train_texts, vocab, batch_size=2)
    val_dataloader = prepare_language_modeling_data(val_texts, vocab, batch_size=2)
    
    device = get_device()
    print(f"\nUtilisation de: {device}")
    
    # === Partie 1: Comparaison RNN vs LSTM vs GRU ===
    print("\n" + "="*60)
    print("Comparaison RNN / LSTM / GRU pour le modèle de langage")
    print("="*60)
    
    rnn_types = ['rnn', 'lstm', 'gru']
    results = []
    
    embedding_dim = 64
    hidden_dim = 128
    num_layers = 1
    epochs = 15
    
    for rnn_type in rnn_types:
        print(f"\n--- Entraînement {rnn_type.upper()} ---")
        model = LanguageModel(len(vocab), embedding_dim, hidden_dim, num_layers, rnn_type).to(device)
        result = train_language_model_epochs(
            model, train_dataloader, val_dataloader, len(vocab), device, epochs=epochs
        )
        results.append({
            'Type': rnn_type.upper(),
            'Perplexité finale': f"{result['final_val_perplexity']:.2f}",
            'Temps par époque (s)': f"{result['avg_epoch_time']:.3f}",
            'Meilleure perte': f"{result['best_val_loss']:.4f}"
        })
        
    # Afficher le tableau comparatif
    print("\n" + "="*60)
    print("Tableau comparatif des architectures")
    print("="*60)
    print(tabulate(results, headers='keys', tablefmt='grid'))
    
    # === Partie 2: Entraînement du Seq2Seq ===
    print("\n" + "="*60)
    print("Entraînement Seq2Seq")
    print("="*60)
    
    # Préparer les données Seq2Seq
    seq2seq_dataloader = prepare_seq2seq_data(texts, vocab, batch_size=2)
    
    # Créer le modèle Seq2Seq
    encoder = Encoder(len(vocab), embedding_dim, hidden_dim, num_layers, 'gru').to(device)
    decoder = Decoder(len(vocab), embedding_dim, hidden_dim, num_layers, 'gru').to(device)
    seq2seq_model = Seq2Seq(encoder, decoder, device).to(device)
    
    # Entraîner Seq2Seq
    import torch.nn as nn
    import torch.optim as optim
    from train import train_seq2seq, evaluate_seq2seq
    
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.token2idx['<PAD>'])
    optimizer = optim.Adam(seq2seq_model.parameters(), lr=0.001)
    seq2seq_epochs = 30
    clip = 1.0
    
    best_seq2seq_loss = float('inf')
    for epoch in range(seq2seq_epochs):
        train_loss = train_seq2seq(seq2seq_model, seq2seq_dataloader, criterion, optimizer, device, clip)
        val_loss = evaluate_seq2seq(seq2seq_model, seq2seq_dataloader, criterion, device)
        val_perplexity = torch.exp(torch.tensor(val_loss)).item()
        print(f"Époque {epoch+1}/{seq2seq_epochs} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | Perplexité: {val_perplexity:.2f}")
        if val_loss < best_seq2seq_loss:
            best_seq2seq_loss = val_loss
    
    # === Partie 3: Test des décodages ===
    print("\n" + "="*60)
    print("Test des décodages Greedy et Beam Search")
    print("="*60)
    
    # Prendre une phrase de test
    test_text = random.choice(texts)
    print(f"\nPhrase d'origine: {test_text}")
    
    src_seq = vocab.text_to_sequence(test_text, add_eos=True, add_sos=True)
    src_tensor = torch.tensor([src_seq], dtype=torch.long).to(device)
    
    # Greedy Decode
    greedy_tokens = greedy_decode(seq2seq_model, vocab, src_tensor, device=device)
    greedy_text = vocab.sequence_to_text(greedy_tokens)
    print(f"Décodage Glouton: {greedy_text}")
    
    # Beam Search
    beam_tokens = beam_search_decode(seq2seq_model, vocab, src_tensor, k=3, device=device)
    beam_text = vocab.sequence_to_text(beam_tokens)
    print(f"Décodage Beam Search (k=3): {beam_text}")


if __name__ == "__main__":
    main()
