"""
Script principal — Partie III : modèles séquentiels et Seq2Seq.

Utilise un corpus réel de traduction anglais -> français (Tatoeba fra-eng,
téléchargé automatiquement, cf. data_prep.py). Le modèle de langage
(comparaison RNN/LSTM/GRU) est entraîné sur les phrases anglaises de ce même
corpus réel ; le système Seq2Seq effectue une véritable traduction
anglais -> français (et non plus un simple auto-encodage de phrases).
"""

import random

import torch
import torch.nn as nn
import torch.optim as optim
from tabulate import tabulate

from data_prep import (
    Vocabulary, SPECIAL_TOKENS,
    read_fra_eng_pairs, filter_pairs_by_length,
    prepare_language_modeling_data, prepare_translation_data,
)
from models import LanguageModel, Encoder, Decoder, Seq2Seq, greedy_decode, beam_search_decode
from train import (
    get_device, train_language_model_epochs,
    train_seq2seq, evaluate_seq2seq, evaluate_bleu_on_dataset,
)


# ----------------------------------------------------------------------
# Hyperparamètres
# ----------------------------------------------------------------------
NUM_EXAMPLES = 20000       # taille max de corpus brut lu avant filtrage
MAX_SENT_LEN = 10          # longueur max (en tokens) pour garder l'entraînement rapide
MIN_FREQ = 2                # fréquence min pour qu'un mot entre dans le vocabulaire
EMBEDDING_DIM = 64
HIDDEN_DIM = 128
NUM_LAYERS = 1
LM_EPOCHS = 10
SEQ2SEQ_EPOCHS = 20
BATCH_SIZE = 32


def main():
    print("=" * 70)
    print("Partie III : Modèles de langage et Seq2Seq (corpus réel Tatoeba fra-eng)")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1) Chargement du corpus réel + filtrage par longueur
    # ------------------------------------------------------------------
    pairs = read_fra_eng_pairs(num_examples=NUM_EXAMPLES)
    pairs = filter_pairs_by_length(pairs, max_len=MAX_SENT_LEN)
    random.shuffle(pairs)
    print(f"\nNombre de paires (en, fr) après filtrage (<= {MAX_SENT_LEN} tokens) : {len(pairs)}")

    # Split train / val / test (80 / 10 / 10), comme pour les autres parties du projet
    n = len(pairs)
    n_train = max(int(0.8 * n), 1)
    n_val = max(int(0.1 * n), 1)
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:n_train + n_val]
    test_pairs = pairs[n_train + n_val:] or val_pairs  # garde-fou si corpus très petit
    print(f"Train: {len(train_pairs)} | Val: {len(val_pairs)} | Test: {len(test_pairs)}")

    # ------------------------------------------------------------------
    # 2) Construction des vocabulaires source (EN) et cible (FR)
    # ------------------------------------------------------------------
    src_vocab = Vocabulary(SPECIAL_TOKENS)
    trg_vocab = Vocabulary(SPECIAL_TOKENS)
    src_vocab.build_vocab([en for en, fr in train_pairs], min_freq=MIN_FREQ)
    trg_vocab.build_vocab([fr for en, fr in train_pairs], min_freq=MIN_FREQ)
    print(f"\nTaille vocabulaire source (EN) : {len(src_vocab)}")
    print(f"Taille vocabulaire cible (FR)  : {len(trg_vocab)}")

    device = get_device()
    print(f"\nUtilisation de : {device}")

    # ------------------------------------------------------------------
    # 3) Modèle de langage : comparaison RNN / LSTM / GRU
    #    (prédiction du prochain token sur les phrases anglaises du corpus réel)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Comparaison RNN / LSTM / GRU — modèle de langage (corpus réel, anglais)")
    print("=" * 70)

    lm_train_texts = [en for en, fr in train_pairs]
    lm_val_texts = [en for en, fr in val_pairs]
    lm_train_loader = prepare_language_modeling_data(lm_train_texts, src_vocab, batch_size=BATCH_SIZE)
    lm_val_loader = prepare_language_modeling_data(lm_val_texts, src_vocab, batch_size=BATCH_SIZE)

    rnn_types = ['rnn', 'lstm', 'gru']
    lm_results = []

    for rnn_type in rnn_types:
        print(f"\n--- Entraînement {rnn_type.upper()} ---")
        model = LanguageModel(len(src_vocab), EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, rnn_type).to(device)
        result = train_language_model_epochs(
            model, lm_train_loader, lm_val_loader, len(src_vocab), device, epochs=LM_EPOCHS
        )
        lm_results.append({
            'Type': rnn_type.upper(),
            'Perplexité finale (val)': f"{result['final_val_perplexity']:.2f}",
            'Temps/époque (s)': f"{result['avg_epoch_time']:.3f}",
            'Meilleure perte (val)': f"{result['best_val_loss']:.4f}",
        })

    print("\nTableau comparatif RNN / LSTM / GRU :")
    print(tabulate(lm_results, headers='keys', tablefmt='grid'))

    # ------------------------------------------------------------------
    # 4) Système Seq2Seq : véritable traduction EN -> FR
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Entraînement Seq2Seq (traduction anglais -> français, corpus réel)")
    print("=" * 70)

    train_loader = prepare_translation_data(train_pairs, src_vocab, trg_vocab, batch_size=BATCH_SIZE)
    val_loader = prepare_translation_data(val_pairs, src_vocab, trg_vocab, batch_size=BATCH_SIZE)

    encoder = Encoder(len(src_vocab), EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, 'gru').to(device)
    decoder = Decoder(len(trg_vocab), EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, 'gru').to(device)
    seq2seq_model = Seq2Seq(encoder, decoder, device).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=trg_vocab.token2idx['<PAD>'])
    optimizer = optim.Adam(seq2seq_model.parameters(), lr=0.001)
    clip = 1.0

    # Étude de l'effet du gradient clipping : on entraîne 3 époques sans clip puis
    # on observe la norme du gradient avant/après pour illustrer l'intérêt du clipping.
    print("\n--- Illustration de l'effet du gradient clipping (3 époques tests) ---")
    for epoch in range(3):
        for src, trg in train_loader:
            src, trg = src.to(device), trg.to(device)
            optimizer.zero_grad()
            outputs = seq2seq_model(src, trg)
            output_dim = outputs.shape[-1]
            loss = criterion(outputs[:, 1:].reshape(-1, output_dim), trg[:, 1:].reshape(-1))
            loss.backward()
            grad_norm_before = torch.nn.utils.clip_grad_norm_(seq2seq_model.parameters(), float('inf'))
            torch.nn.utils.clip_grad_norm_(seq2seq_model.parameters(), clip)
            optimizer.step()
            break  # un seul batch par époque test, à but illustratif
        print(f"  Époque test {epoch+1} : norme du gradient avant clipping = {grad_norm_before:.3f} "
              f"(seuil de clipping = {clip})")

    print("\n--- Entraînement complet du Seq2Seq ---")
    best_val_loss = float('inf')
    for epoch in range(SEQ2SEQ_EPOCHS):
        train_loss = train_seq2seq(seq2seq_model, train_loader, criterion, optimizer, device, clip)
        val_loss = evaluate_seq2seq(seq2seq_model, val_loader, criterion, device)
        val_perplexity = torch.exp(torch.tensor(val_loss)).item()
        print(f"Époque {epoch+1}/{SEQ2SEQ_EPOCHS} | Train loss: {train_loss:.4f} | "
              f"Val loss: {val_loss:.4f} | Perplexité: {val_perplexity:.2f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(seq2seq_model.state_dict(), "best_seq2seq.pth")

    seq2seq_model.load_state_dict(torch.load("best_seq2seq.pth"))

    # ------------------------------------------------------------------
    # 5) Évaluation BLEU sur le jeu de test réel
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Évaluation BLEU sur le jeu de test (corpus réel)")
    print("=" * 70)
    bleu_score = evaluate_bleu_on_dataset(
        seq2seq_model, test_pairs, src_vocab, trg_vocab, device, num_samples=min(50, len(test_pairs))
    )
    print(f"Score BLEU moyen (glouton, échantillon test) : {bleu_score:.4f}")

    # ------------------------------------------------------------------
    # 6) Comparaison décodage glouton vs beam search sur des phrases réelles
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Comparaison décodage Glouton vs Beam Search (phrases réelles du test set)")
    print("=" * 70)

    examples = random.sample(test_pairs, min(5, len(test_pairs)))
    decode_table = []
    for en, fr in examples:
        src_seq = src_vocab.text_to_sequence(en, add_eos=True, add_sos=False)
        src_tensor = torch.tensor([src_seq], dtype=torch.long).to(device)

        greedy_tokens = greedy_decode(seq2seq_model, trg_vocab, src_tensor, device=device)
        greedy_text = trg_vocab.sequence_to_text(greedy_tokens)

        beam_tokens = beam_search_decode(seq2seq_model, trg_vocab, src_tensor, k=3, device=device)
        beam_text = trg_vocab.sequence_to_text(beam_tokens)

        decode_table.append({
            "Source (EN)": en,
            "Référence (FR)": fr,
            "Glouton": greedy_text,
            "Beam (k=3)": beam_text,
        })

    print(tabulate(decode_table, headers='keys', tablefmt='grid'))

    print("\nTerminé. Voir best_seq2seq.pth pour le modèle entraîné.")


if __name__ == "__main__":
    main()
