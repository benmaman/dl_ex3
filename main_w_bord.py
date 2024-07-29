import pandas as pd
import re
from gensim.models import Word2Vec
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from models import LyricsGenerator, LyricsDataset, generate_text, MergeLyricsGenerator
import os
from sklearn.model_selection import train_test_split
import mido
from tools import low_case_name_file, generate_sequences, find_exact_index
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np

# editor_parameter
embedding_dim = 300  # entities per word
hidden_dim = 40
num_layers = 2
batch_size = 10
sequence_length = 5  # length of the input of the model
num_epochs = 10
midi_dim = 50

# TensorBoard SummaryWriter
writer = SummaryWriter('runs/lyrics_generation_experiment')

# 1. Load your dataset
data = pd.read_csv('lyrics_train_set.csv')
data = data.iloc[:, :]
sentences = []

midi_folder = "midi_files"
melody_embeddings = []

# add the embedding of midi files
midi_embedding = pd.read_csv('matched_embeddings.csv')
data = data.merge(midi_embedding, on=['singer', 'song'])

# 2. Embedding
# pre-processing
for lyrics in data['lyrics']:
    # Lowercase and remove characters between parentheses
    cleaned_lyrics = lyrics.lower()
    cleaned_lyrics = re.sub(r'\(.*?\)', '', cleaned_lyrics)

    # Split into words, then add 'eos' at the end
    words = cleaned_lyrics.split()
    words.append('eos')  # Append 'eos' as the end-of-sentence marker for the entire lyrics
    sentences.append(words)
    ### tbd- add padding

# word2vec
word2vec = Word2Vec(sentences, vector_size=embedding_dim, window=5, min_count=1, workers=4)
word2vec.train(sentences, total_examples=len(sentences), epochs=10)

word_to_idx = {word: idx for idx, word in enumerate(word2vec.wv.index_to_key)}
idx_to_word = {idx: word for word, idx in word_to_idx.items()}
weights = torch.FloatTensor(word2vec.wv.vectors)

train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)

# Generate sequences for training set
train_input_vectors, train_target_indices, train_midi_sequences = generate_sequences(
    train_data, sequence_length, word2vec, word_to_idx)

# Generate sequences for validation set
val_input_vectors, val_target_indices, val_midi_sequences = generate_sequences(
    val_data, sequence_length, word2vec, word_to_idx)



train_dataset = LyricsDataset(train_input_vectors, train_target_indices, train_midi_sequences)
val_dataset = LyricsDataset(val_input_vectors, val_target_indices, val_midi_sequences)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Model, loss function, and optimizer
vocab_size = len(word_to_idx)
model = MergeLyricsGenerator(vocab_size, embedding_dim, hidden_dim, num_layers, midi_dim)
model.embedding.weight = nn.Parameter(weights)
model.embedding.weight.requires_grad = False  # Freeze the embeddings

# Dummy input tensor for visualizing the model graph in TensorBoard
dummy_input = (torch.zeros(batch_size, sequence_length, embedding_dim), 
                torch.zeros(batch_size, midi_dim),
                model.init_hidden(batch_size))


writer.add_graph(model, dummy_input)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for input_batch, target_batch, midi_batch in train_loader:
        current_batch_size = input_batch.size(0)
        hidden = model.init_hidden(current_batch_size)

        optimizer.zero_grad()
        output, hidden = model(input_batch, midi_batch, hidden)
        output = output.view(-1, vocab_size)
        loss = criterion(output, target_batch.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        hidden = tuple([h.detach() for h in hidden])
    
    avg_training_loss = total_loss / len(train_loader)
    train_perplexity = np.exp(avg_training_loss)
    print(f'Epoch {epoch + 1}, Training Loss: {avg_training_loss}, Training Perplexity: {train_perplexity}')

    # Log training loss to TensorBoard
    writer.add_scalar('Loss/Training', avg_training_loss, epoch)
    writer.add_scalar('Perplexity/Training', train_perplexity, epoch)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for input_batch, target_batch, midi_batch in val_loader:
            current_batch_size = input_batch.size(0)
            hidden = model.init_hidden(current_batch_size)
            output, hidden = model(input_batch, midi_batch, hidden)
            output = output.view(-1, vocab_size)
            loss = criterion(output, target_batch.view(-1))
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    val_perplexity = np.exp(avg_val_loss)
    print(f'Epoch {epoch + 1}, Validation Loss: {avg_val_loss}, Validation Perplexity: {val_perplexity}')

    # Log validation loss to TensorBoard
    writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
    writer.add_scalar('Perplexity/Validation', val_perplexity, epoch)

    seed_text = "another"
    last_midi_embedding = val_midi_sequences[-1]
    song, singer = find_exact_index(data, last_midi_embedding)
    print(f"for melody: {song}, {singer}:")

    generated_text = generate_text(seed_text, model, sequence_length, 50, vocab_size, word_to_idx, idx_to_word, word2vec, last_midi_embedding)
    print(f"Generated Text after Epoch {epoch + 1}: {generated_text}")

# Close the TensorBoard writer
writer.close()
