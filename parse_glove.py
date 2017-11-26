import numpy as np
import pickle
import random

glove_dir = "./glove.6B.100d.txt"

def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb+') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def get_glove():
    dictionary = {}
    counter = 0
    ids_to_words = {}
    words_to_ids = {}
    embed_matrix = []
    try:
        dictionary = load_obj('glove_word_to_vector')
        ids_to_words = load_obj('glove_ids_to_words')
        words_to_ids = load_obj('glove_words_to_ids')
        embed_matrix = load_obj('glove_embed_matrix')
        
        print("Loading Glove vectors from Saved files....")
    except:
        with open (glove_dir, 'r') as f:
            for line in f:
                line = line.strip()
                line = line.split()
                vector = np.array([float(x) for x in line[1:]])
                dictionary[line[0]] = vector
                ids_to_words[counter] = line[0]
                words_to_ids[line[0]] = counter
                counter +=1
                embed_matrix.append(vector)
        embed_matrix = np.asarray(embed_matrix)

        print("created new glove vectors....")
    return dictionary, ids_to_words, words_to_ids, embed_matrix


