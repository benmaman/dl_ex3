import os
import re
import numpy 
from models import *

def preprocessing_lyrics(data):
    """ Initial preprocessing 

    Args:
        data (Dataframe): contain lyrics column

    Returns:
        sequence (list) : list, eac object contain cleaned and processesd lyrics
    """
    sentences=[]
    for lyrics in data['lyrics']:
        # Lowercase and remove characters between parentheses
        cleaned_lyrics = lyrics.lower()
        cleaned_lyrics = re.sub(r'\(.*?\)', '', cleaned_lyrics)
        
        # Split into words, then add 'eos' at the end
        words = cleaned_lyrics.split()
        words.append('eos')  # Append 'eos' as the end-of-sentence marker for the entire lyrics
        sentences.append(words)
    return sentences

def generate_sequences(data, sequence_length, word2vec, word_to_idx):
    input_sequences = []
    target_sequences = []
    midi_sequences = []

    for _, row in data.iterrows():
        lyrics = row['lyrics']
        midi_vector = row.iloc[-50:].values.tolist()  # Get the MIDI embedding for the current song
        
        # Lowercase and remove characters between parentheses
        cleaned_lyrics = lyrics.lower()
        cleaned_lyrics = re.sub(r'\(.*?\)', '', cleaned_lyrics)
        
        # Split into words and add 'eos' at the end
        words = cleaned_lyrics.split()
        words.append('eos')

        if len(words) >= sequence_length:
            for i in range(len(words) - sequence_length):
                input_sequences.append(words[i:i + sequence_length])
                target_sequences.append(words[i + 1:i + sequence_length + 1])
                midi_sequences.append(midi_vector)

    input_vectors = [[word2vec.wv[word] for word in sequence] for sequence in input_sequences]
    target_indices = [[word_to_idx[word] for word in sequence] for sequence in target_sequences]

    return input_vectors, target_indices, midi_sequences



def generate_text(seed_text, model,sequence_length, max_length, vocab_size, word_to_idx, idx_to_word, word2vec, midi_embedding):
    """
    Generates text using the trained LSTM model starting from a seed text.
    
    Args:
        seed_text (str): Initial text to start the generation.
        model (LyricsGenerator): Trained lyrics generation model.
        max_length (int): Maximum length of the generated text.
        vocab_size (int): Size of the vocabulary.
        word_to_idx (dict): Dictionary mapping words to their indices.
        idx_to_word (dict): Dictionary mapping indices to their words.
        word2vec (gensim.models.Word2Vec): Pre-trained Word2Vec model.
        midi_embedding (numpy.ndarray): MIDI embedding to use during generation.
        
    Returns:
        str: Generated text.
    """
    model.eval()
    words = seed_text.lower().split()
    input_indices = [word2vec.wv[word] for word in words]
    x = torch.tensor([input_indices], dtype=torch.float32)  # Add batch dimension

    midi = torch.tensor(midi_embedding, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

    state_h, state_c = model.init_hidden(1)  # Initialize hidden state with batch size 1

    generated_words = words.copy()
    for _ in range(max_length):
        y_pred, (state_h, state_c) = model(x, midi, (state_h, state_c))
        last_word_logits = y_pred[0, -1]  # Get the last time step
        p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
        word_idx = np.random.choice(len(last_word_logits), p=p)
        new_word = idx_to_word[word_idx]
        generated_words.append(new_word)

        # Update x for the next prediction
        input_text = generated_words[-sequence_length:]
        input_indices = [word2vec.wv[word] for word in input_text]
        x = torch.tensor([input_indices], dtype=torch.float32)  # Add batch dimension

    # Edit the final lyrics - take the first eos or last &
    final_lyrics = []
    if 'eos' in generated_words:
        # Find the index of 'eos' and take all words before it
        index = generated_words.index('eos')
        final_lyrics=generated_words[:index]
        return ' '.join(final_lyrics)
    elif '&' in generated_words:
        # If there is no 'eos', take all the words
        last_index = len(generated_words) - 1 - generated_words[::-1].index('&')
        final_lyrics=generated_words[:last_index]
    else:
        final_lyrics=generated_words

    return ' '.join(final_lyrics)

def low_case_name_file(midi_folder):
    """rename the name of the folder in order to import the name of the files.

    Args:
        midi_folder (_type_): _description_

    Returns:
        _type_: _description_
    """
    # List all files in the folder
    files = os.listdir(midi_folder)

    # Loop over all files in the folder
    for file in files:
        # Check if the file is a MIDI file
        if file.endswith('.mid'):
            # Get the current file path
            old_file_path = os.path.join(midi_folder, file)
            
            # Convert the file name to lowercase
            new_file_name = file.lower()
            new_file_path = os.path.join(midi_folder, new_file_name)
            
            # Rename the file
            os.rename(old_file_path, new_file_path)
            print(f"Renamed '{file}' to '{new_file_name}'")

    return print("All MIDI files have been renamed to lowercase.")


def find_exact_index(df, midi_embedding):

    """
    Find the exact index in the DataFrame that matches the given MIDI embedding.

    Args:
        df (pd.DataFrame): DataFrame containing the MIDI embeddings.
        midi_embedding (list): List containing the 50-column MIDI embedding.

    Returns:
        list: List of indices that match the given MIDI embedding.
    """
    # Create a boolean mask for rows that match the MIDI embedding
    mask = (df.iloc[:, -50:] == midi_embedding).all(axis=1)
    
    # Get the indices of the matching rows
    matching_indices = df[mask].index.tolist()
    song,singer=df.iloc[matching_indices,0].values[0],df.iloc[matching_indices,1].values[0]

    return song,singer