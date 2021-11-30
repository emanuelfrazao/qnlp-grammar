from typing import NamedTuple, List, Tuple, Iterable
import numpy as np

from .representation.tensors import ExtensionalTensorizer



def preprocess(instances: 'Iterable[Iterable[str]]',
               tensorizer: 'ExtensionalTensorizer') -> 'Tuple[np.ndarray, np.ndarray]':
    """Turns data into arrays.

    Args:
        instances (Iterable[Iterable[str]]): An iterable of instances of the form (sentence, sentence, label).
        tensorizer (ExtensionalTensorizer): A trained tensorizer object.

    Returns:
        np.ndarray: The features array, as (n_instances x n_sentences x space_dimension).
    """
    features = []
    labels = []
    for (sentence1, sentence2, label) in instances:
        # Tensorize sentences and save them
        features.append(np.stack((tensorizer.tensorize(sentence1), tensorizer.tensorize(sentence2)), axis=1).T)
        # Save labels
        labels.append(int(label))

    # Turn them into arrays
    features = np.stack(features, axis=0)
    labels = np.array(labels).reshape(-1, 1)

    # Normalize each sentence in features
    features = features / np.linalg.norm(features, axis=2, keepdims=True)
    
    return features, labels
    


def get_data_from_source(data_path: 'str') -> 'List[str]':
    
    with open(data_path, 'r') as f:
        content = f.readlines()
    
    return [instance.replace('\n', '').split(',') for instance in content]


def train_test_split(data: 'List[str]',
                     train_ratio: 'float'=.7,
                     test_ratio: 'float' =.15, 
                     shuffle: 'bool'=True):
 
    if shuffle:
        working_data = data.copy()
        np.random.shuffle(working_data)
    else:
        working_data = data
        
    n_total = len(working_data)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * (1 - train_ratio - test_ratio))
    
    return working_data[:n_train], working_data[n_train:n_train + n_val], working_data[n_train + n_val:]
    

def get_sentences(data: 'List[str]'):
    
    return [sentence for instance in data for sentence in instance[:2]] 