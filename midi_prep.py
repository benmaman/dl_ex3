"""
in this script i will use the pretty midi library to imbed the midi files
in diffrent ways so it can be used for the LSTM model

"""

# %%-----------------------------imports---------------------------------------
import pretty_midi
import numpy as np
import os
import sys
import random
from tqdm import tqdm
import matplotlib.pyplot as plt 
import librosa.display
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


# %%-----------------------------functions-------------------------------------

def plot_piano_roll(pm, start_pitch, end_pitch, fs=100):
    """
    Plot the piano roll of a PrettyMIDI object.

    Parameters:
    - pm (PrettyMIDI): The PrettyMIDI object to plot.
    - start_pitch (int): The starting pitch index to display.
    - end_pitch (int): The ending pitch index to display.
    - fs (int, optional): The sampling rate in Hz. Default is 100.

    Returns:
    None
    """
    # Use librosa's specshow function for displaying the piano roll
    librosa.display.specshow(pm.get_piano_roll(fs)[start_pitch:end_pitch],
                             hop_length=1, sr=fs, x_axis='time', y_axis='cqt_note',
                             fmin=pretty_midi.note_number_to_hz(start_pitch))




def wpm_scaler(tempo, time_of_song, num_words):
    """
    Use the tempo to estimate words in the song.
    
    Parameters:
    - tempo (float): The tempo of the song (beats per minute).
    - time_of_song (float): The duration of the song in seconds.
    - num_words (int): The actual number of words in the song.
    
    Returns:
    - float: Estimated number of words in the song.
    
    Raises:
    - ValueError: If the estimated words differ significantly from the actual words.
    """
    # Create a scaler
    scaler = MinMaxScaler(feature_range=(60, 300))  # Adjust range as needed
    scaler.fit(np.array([[30], [280]]))  # Min tempo is 30, max tempo is 280
    
    # Estimate words per minute based on tempo
    estimated_wpm = scaler.transform([[tempo]])[0][0]
    # print(f'Estimated WPM: {estimated_wpm}')
    # Estimate the number of words in the song
    
    estimated_words = estimated_wpm * (time_of_song/60)
    # print(f'Estimated words: {estimated_words}')
    # print(f'Actual words: {num_words}')
    
    # Raise error if the estimated words are significantly different from the actual words
    if abs(estimated_words - num_words) > 10000:
        raise ValueError('The estimated words differ significantly from the actual words')
    
    return estimated_words


def max_pooling_to_size(embedding, desired_length):
    """
    Applies max pooling to an embedding array to achieve the desired length.

    Parameters:
    embedding (numpy.ndarray): The input embedding array.
    desired_length (int): The desired length of the pooled embedding.

    Returns:
    numpy.ndarray: The pooled embedding array with the desired length.

    Raises:
    ValueError: If the desired length cannot be achieved due to rounding issues.
    """

    original_length = len(embedding)
    
    # Calculate the pool size
    pool_size = int(np.ceil(original_length / desired_length))
    
    # Pad the embedding if it's not perfectly divisible by pool_size
    if original_length % pool_size != 0:
        padding_length = pool_size - (original_length % pool_size)
        embedding = np.pad(embedding, (0, padding_length), mode='constant', constant_values=-np.inf)
    
    # Reshape the embedding to have the pool_size as the second dimension
    reshaped_embedding = embedding.reshape(-1, pool_size)
    
    # Apply max pooling
    pooled_embedding = np.max(reshaped_embedding, axis=1)
    
    # Ensure the final length matches the desired length
    if len(pooled_embedding) > desired_length:
        pooled_embedding = pooled_embedding[:desired_length]
    elif len(pooled_embedding) < desired_length:
        # Pad the embedding with zeros
        padding_length = desired_length - len(pooled_embedding)
        pooled_embedding = np.pad(pooled_embedding, (0, padding_length), mode='constant', constant_values=0)
        
    return pooled_embedding


# %%-----------------------------data prep--------------------------------


# %%-----------------------------main-------------------------------------------


def main_midi(file,trainsongs, testsongs, midi_data,dim_midi, verbose=False):

    artist, song = file.split('_-_')[0], file.split('_-_')[1].split('.mid')[0]

    artist = artist.replace('_', ' ').lower()
    song = song.replace('_', ' ').lower()

    # get the number of words in the lyrics
    try:
        lyrics = trainsongs[(trainsongs['singer'] == artist) & (trainsongs['song'] == song)]['lyrics']
    except  KeyError:
        try:
            lyrics = testsongs[(testsongs['singer'] == artist) & (testsongs['song'] == song)]['lyrics']
        except KeyError:
            print('song not found')
            return
    
    
    # count the number of words in the lyrics not including & and -
    lyrics = lyrics.str.replace('&', ' ').str.replace('-', ' ').str.replace('  ', ' ').str.split(' ')
    lyrics = [word for sentence in lyrics for word in sentence]
    num_words = len(lyrics)

    # get time of the song
    time_of_s = midi_data.get_end_time()

    # get bpm of the song
    bpm = midi_data.estimate_tempo()

    # estimate the number of words in the song
    estemat_words = wpm_scaler(bpm, time_of_s, num_words)

    # devide the song into the number of estemat_words
    time_of_word = (time_of_s/60) / estemat_words

    # get piolo roll
    midi_features = midi_data.get_piano_roll(fs=1/ time_of_word)
    if verbose:
        print(f"midi_features shape: {midi_features.shape}")
        # plot_piano_roll
        plt.figure(figsize=(12, 4))
        plot_piano_roll(midi_data, 0, 128, fs=1/ time_of_word)
        plt.show()

    # turn the midi_features into 1D array by taking the mode of each column
    # midi_features = np.argmax(midi_features, axis=0)
    # turn the midi_features into 1D array by taking the variance of each column
    midi_features = np.var(midi_features, axis=0)

    if verbose:
        # plot the midi_features
        plt.figure(figsize=(12, 4))
        plt.plot(midi_features)
        plt.show()

    # Example embedding vector
    embedding = midi_features



    # Apply max pooling to achieve the desired length
    compressed_embedding = max_pooling_to_size(embedding, dim_midi)
    if verbose:
        print("Original embedding:", embedding)
        print("Compressed embedding:", compressed_embedding)

        plt.figure(figsize=(12, 4))
        plt.plot(compressed_embedding)
        plt.show()

    return compressed_embedding, artist, song



def midi_embed(embedding_dim,verbose=False):
    """
    Embeds MIDI files into a DataFrame.
    Args:
        embedding_dim (int): The dimension of the embedding vector.
        train_songs (pd.DataFrame): The DataFrame containing the training songs data.
        test_songs (pd.DataFrame): The DataFrame containing the test songs data.
        midi_data (pretty_midi.PrettyMIDI): The MIDI data to be embedded.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
    Returns:
        pd.DataFrame: The DataFrame containing the embedded MIDI data.
    """
    
    train_songs = pd.read_csv('lyrics_train_set.csv')
    test_songs = pd.read_csv('lyrics_test_set.csv')


    # Load MIDI file into PrettyMIDI object
    # midi_data = pretty_midi.PrettyMIDI('C:/Users/sacha/Code/dl_ex3/midi_files/abba_-_knowing_me_knowing_you.mid')

    # file = '2_unlimited_-_get_ready_for_this.mid'

    columns = list(range(1,embedding_dim+1)) + ['singer', 'song']

    midi_df_embedding =  pd.DataFrame(columns=columns)

    # loop through the midi files
    for file in os.listdir('midi_files'):
        if file.endswith('.mid'):
            try:
                print(f'Processing {file}')
                midi_data = pretty_midi.PrettyMIDI('midi_files/' + file)
            except:
                # in red
                red = "\033[1;31m"
                print(f'{red} Error loading {file}')
                # stop the red
                print("\033[0m")
                    
                continue
            imbed_vec, artis_name, song_name = main_midi(file, train_songs, test_songs, midi_data, embedding_dim,verbose=verbose)

            row_data = list(imbed_vec) + [artis_name, song_name]
        
            # Convert to Series with correct index, then append
            row_series = pd.Series(row_data, index=columns).to_frame().T
            midi_df_embedding = pd.concat([midi_df_embedding,row_series])


    # save the midi_df_embedding
    # change the coloms to start from 1
    
    midi_df_embedding.to_csv(os.getcwd()+'/sinai_matched_embeddings.csv', index=False)
    return midi_df_embedding


if __name__ == '__main__':
    midi_embed(50, verbose=True)
