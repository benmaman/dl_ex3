import pandas as pd
import re
from gensim.models import Word2Vec
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from models import LyricsGenerator, LyricsDataset, MergeLyricsGenerator
import os
from sklearn.model_selection import train_test_split
import mido
from tools import low_case_name_file,generate_sequences,find_exact_index,generate_text,preprocessing_lyrics,generate_text_gpu,preprocessing_lyrics_back_to_df
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from datetime import datetime
import numpy as np
import itertools
from more_eval import text_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from torch.utils.tensorboard import SummaryWriter
from midi_prep import midi_embed
import time
# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# best hyperparameters
"""
1. {epoch : 15, hidden dimensions : 40, lstm layer : 2, batch size: 16, 
    sequence lenght: 5, learning rate : 0.0001,dropout:0.1, 
    model:  merge, midi embedding: graph}
2. {epoch : 7, hidden dimensions : 40, lstm layer : 2, batch size: 16,
     sequence lenght: 5, learning rate : 0.001
    ,dropout:0.1, model:  naive, midi embedding: modified}

"""




param_grid={'batch_size':[16],
            'sequence_length':[5],
            'learning_rate':[0.001,0.0001],
            'dropout':[0.1],
            'embedding_dim':[300],
            'hidden_dim':[40],
            'num_layers':[2],
            'midi_dim':[50],
            'model_name':['naive','merge'],
            'midi_method':['modified','graph'],
            'num_epochs':[7,15]

}

results_df = pd.DataFrame()
i=1
param_combinations = itertools.product(
    param_grid['batch_size'],
    param_grid['sequence_length'],
    param_grid['learning_rate'],
    param_grid['dropout'],
    param_grid['embedding_dim'],
    param_grid['hidden_dim'],
    param_grid['num_layers'],
    param_grid['midi_dim'],
    param_grid['model_name'],
    param_grid['midi_method'],
    param_grid['num_epochs']

)

# Grid search loop using itertools
for batch_size, sequence_length, learning_rate, dropout,embedding_dim, hidden_dim, num_layers, midi_dim, model_name, midi_method ,num_epochs in param_combinations:
    # TensorBoard SummaryWriter
    name = 'learning_rate='+str(learning_rate)+'model_name='+model_name+'midi_method='+midi_method+'num_epochs='+str(num_epochs)
    writer = SummaryWriter('runs/'+name)
    print(f"start itrer numer {i}")
    i+=1
    #editor_parameter
    # embedding_dim = 300 #entities per word
    # hidden_dim = 40
    # num_layers = 2
    # sequence_length=sequence_length #lenght of the input of the model
    # num_epochs = 15
    # midi_dim = 50
    # learning_rate=learning_rate
    # dropout=dropout
    # model_name='naive' #model configure, can get 'naive' or 'merge'
    # midi_method='modified' # midi configure, can get 'graph' or 'modified'

    #1. Load your dataset
    data = pd.read_csv('lyrics_train_set.csv')
    lyr_data = preprocessing_lyrics_back_to_df(data)
    data=data.iloc[:,:]
    sentences = []

    midi_folder = "midi_files"
    melody_embeddings = []

    #add the embedding of midi files
    if midi_method=='graph':
        midi_embedding = pd.read_csv('matched_embeddings_graph.csv')
    if midi_method=='modified':
        midi_embedding = pd.read_csv('matched_embeddings_modified.csv')
        data_to_normalize = midi_embedding.iloc[:, :-2]
        indices = midi_embedding.iloc[:, -2:]

        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(data_to_normalize)

        # Combine the normalized data with the index columns
        midi_embedding_normalized = pd.DataFrame(normalized_data, columns=data_to_normalize.columns)
        midi_embedding = pd.concat([midi_embedding_normalized, indices], axis=1)
    data=data.merge(midi_embedding,on=['singer',	'song'])


    ## 2.Embedding
    
    # pre-processing 
    sentences=preprocessing_lyrics(data)

    #word2vect
    word2vec = Word2Vec(sentences, vector_size=embedding_dim, window=5, min_count=1, workers=4)
    word2vec.train(sentences, total_examples=len(sentences), epochs=10)
    word2vec.save('results/word2vec.model')
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




    ## 3.ml model
    #split the dataset to train-validation

    # Create dataset and dataloader
    train_dataset = LyricsDataset(train_input_vectors, train_target_indices, train_midi_sequences)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = LyricsDataset(val_input_vectors, val_target_indices, val_midi_sequences)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)




    #Hyper-parameter-tuning
    vocab_size = len(word_to_idx)

    if model_name=='naive':
        model = LyricsGenerator(vocab_size, embedding_dim+50, hidden_dim, num_layers,dropout)
    if model_name=='merge':
        model = MergeLyricsGenerator(vocab_size, embedding_dim, hidden_dim, num_layers,dropout)
    model.embedding.weight = nn.Parameter(weights)
    model.embedding.weight.requires_grad = False  # Freeze the embeddings



    # for tensorboard
    # Dummy input tensor for visualizing the model graph in TensorBoard
    dummy_input = (torch.zeros(batch_size, sequence_length, embedding_dim), 
                    torch.zeros(batch_size, midi_dim),
                    model.init_hidden(batch_size))


    writer.add_graph(model, dummy_input)

    # Move model to GPU
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,weight_decay=1e-5)



    # Initialize lists to store loss and perplexity values
    train_losses = []
    val_losses = []
    train_perplexities = []
    val_perplexities = []
    results={}
    best_val_perplexity = float('inf')
    best_model_state = None
    best_epoch=0
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for input_batch, target_batch, midi_batch in train_loader:
            # gpu
            input_batch, target_batch, midi_batch = input_batch.to(device), target_batch.to(device), midi_batch.to(device)

            current_batch_size = input_batch.size(0)
            hidden = model.init_hidden(current_batch_size)
            # gpu
            hidden = tuple(h.to(device) for h in hidden)


            optimizer.zero_grad()
            output, hidden = model(input_batch, midi_batch, hidden)
            output = output.view(-1, vocab_size)
            loss = criterion(output, target_batch.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            hidden = tuple([h.detach() for h in hidden])

        # Compute training loss and perplexity
        avg_training_loss = total_loss / len(train_loader)
        train_perplexity = np.exp(avg_training_loss)
        train_losses.append(avg_training_loss)
        train_perplexities.append(train_perplexity)
        print(f'Epoch {epoch + 1}, Training Loss: {avg_training_loss}, Training Perplexity: {train_perplexity}')

        # Log training loss to TensorBoard
        writer.add_scalar('Loss/Training', avg_training_loss, epoch)
        writer.add_scalar('Perplexity/Training', train_perplexity, epoch)

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for input_batch, target_batch, midi_batch in val_loader:
                # gpu
                input_batch, target_batch, midi_batch = input_batch.to(device), target_batch.to(device), midi_batch.to(device)


                current_batch_size = input_batch.size(0)
                hidden = model.init_hidden(current_batch_size)
                # gpu
                hidden = tuple(h.to(device) for h in hidden)

                output, hidden = model(input_batch, midi_batch, hidden)
                output = output.view(-1, vocab_size)
                loss = criterion(output, target_batch.view(-1))
                val_loss += loss.item()

        # Compute validation loss and perplexity
        avg_val_loss = val_loss / len(val_loader)
        val_perplexity = np.exp(avg_val_loss)
        val_losses.append(avg_val_loss)
        val_perplexities.append(val_perplexity)
        print(f'Epoch {epoch + 1}, Validation Loss: {avg_val_loss}, Validation Perplexity: {val_perplexity}')

        
        # Log validation loss to TensorBoard
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        writer.add_scalar('Perplexity/Validation', val_perplexity, epoch)

        # Check if this is the best model so far and save it
        if val_perplexity < best_val_perplexity:
            best_val_perplexity = val_perplexity
            best_model_state = model.state_dict()
            best_epoch=epoch
            
        # Generate text after each epoch
        
        last_midi_embedding=val_midi_sequences[3]
        song,singer = find_exact_index(data, last_midi_embedding)
        print(f"for melody: {song},{singer}:")
        # get lyrics of the song
        song_lyrics = lyr_data[(lyr_data['song'] == singer) & (lyr_data['singer'] == song)].lyrics.values[0]
        # first word of the generated text is from the original song
        seed_text = song_lyrics.split(' ')[0]
        print(f"seed text: {seed_text}")

        # generated_text = generate_text(seed_text, model,sequence_length, 50, vocab_size, word_to_idx, idx_to_word, word2vec, last_midi_embedding)
        generated_text = generate_text_gpu(seed_text, model, sequence_length, 50, vocab_size, word_to_idx, idx_to_word, word2vec, last_midi_embedding, device)

        # print('Generated Text:', generated_text)
        # print('Original Text:', song_lyrics)

        # compute the similarity between the generated text and the original text
        similarity_score = text_similarity(song_lyrics[:51], generated_text)
        print(f"Cosine Similarity: {similarity_score}")

        # Log the similarity score to TensorBoard
        writer.add_scalar('Similarity/Generated_Text', similarity_score, epoch)

        print(f"Generated Text after Epoch {epoch + 1}: {generated_text}")

    current_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    results={"date":current_time,
            'epochs':num_epochs,
            'best_epoch':best_epoch + 1,
            'hidden_dim':hidden_dim,
            'lstm_layer':num_layers,
            'batch_size':batch_size,
            'sequence_length':sequence_length,
            'learning_rate':learning_rate,
            'dropout':dropout,
            'model_name':model_name,
            'midi_method':midi_method,
            'train_loss':train_losses,
            'validation_loss':val_losses,
            "train_perplexities":train_perplexities,
            "validation_perplexities":val_perplexities
            }

    # Convert the results dictionary to a DataFrame and append to results_df
    results_df = pd.concat([results_df,pd.DataFrame([results])], ignore_index=True)


    # Close the TensorBoard writer
    writer.close()



# Optionally, save the DataFrame to a file after each epoch or at specific intervals
# Save the best model
current_time=current_time.replace(':',"_")
results_df.to_csv(f'results/training_results_{current_time}.csv', index=False)

torch.save(best_model_state, f'results/best_model_{current_time}.pth')