"""Script for running the base model on training and validation data. 
May be used as a module by providing a Namespace to the main method.
"""
## Modules
import argparse
from pathlib import Path
import numpy as np

# Costum modules path
import sys
sys.path.insert(0, Path(__file__).parent.parent.as_posix())

from modules.preprocessing import preprocess, get_data_from_source, get_sentences, train_test_split
from modules.linguistics.vocab import TypedVocabulary
from modules.linguistics.relations import Context
from modules.representation.tensors import ExtensionalTensorizer
from modules.modelling.models.base import BaseModel



def main(args):
    
    # # Assert available paths
    assert Path(args.train_data).exists() and Path(args.val_data).exists(), f"Data path given does not exist."
    assert Path(args.reference_vocab).exists(), f"Reference vocabulary path given does not exist."   
    assert Path(args.context).exists(), f"Context path given does not exist."   
    
    # # Load data
    train_data = get_data_from_source(args.train_data)
    val_data = get_data_from_source(args.val_data)
    
    ## Train tensor representations on train and validation data
    # Get corpus training
    tensor_corpus = get_sentences(train_data + val_data)
    
    # Get reference vocabulary
    reference_vocab = TypedVocabulary.from_json(args.reference_vocab)
    
    # Get context
    context_base = Context.from_json(args.context)
    
    # Get tensors
    tensorizer = ExtensionalTensorizer(tensor_corpus, reference_vocab, args.minimum_occurences)
    tensorizer.build_tensors(context_base)

    ## Preprocess training and validation data
    X_train, y_train = preprocess(train_data, tensorizer)
    X_val, y_val = preprocess(val_data, tensorizer)    
    
    # # Train model
    base = BaseModel()
    base.fit(X_train, y_train)
    y_pred = base.predict(X_val)

    # # Print results
    acc = lambda true, pred: (true == pred).sum() / true.size
    val_accuracy = acc(y_val.reshape(-1), y_pred)
    print(f"Inner product threshold: {base.threshold:.3f}")
    print(f"{'Higher' if base.higher_is_positive else 'Lower'} inner product pairs correspond to same class.")
    print(f"Accuracy: {val_accuracy*100:.2f}%")
    
    


def arg_parser() -> 'argparse.Namespace':
    """Parses command line arguments.
    
    Returns:
        argparse.Namespace: The object with stored arguments.
    """    
    ## Initializer
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='''
description:
    Script for running the base model on training and validation data.
example:
    python .\model_scripts\\base_training.py 
        --train_data .\data\data_train.txt 
        --val_data .\data\data_val.txt 
        -v .\\vocabs\\reference_vocab.json 
        -c .\contexts\context.json''',
    )

    # # Data
    parser.add_argument(
        '--train_data',
        metavar='<file_path>.txt',
        help='Specify the training data path. Required.',
        type=str,
        required=True
    )
    
    parser.add_argument(
        '--val_data',
        metavar='<file_path>.txt',
        help='Specify the validation data path. Required.',
        type=str,
        required=True
    )

    # Reference vocabulary
    parser.add_argument(
        '-v', '--reference_vocab',
        metavar='<file_path>.json',
        help='Specify the reference vocabulary path. Required.',
        type=str,
        required=True
    )
    
    # Cutoff
    parser.add_argument(
        '-m', '--minimum_occurences',
        metavar='<int>',
        help='Specify the minimum number of occurences in the corpus for the word to have a tensor representation. Defaults to 5.',
        type=int,
        default=5
    )
    
    # Reference vocabulary
    parser.add_argument(
        '-c', '--context',
        metavar='<file_path>.json',
        help='Specify the context basis path. Required.',
        type=str,
        required=True
    )

    # Destination
    parser.add_argument(
        '-dest', '--destination',
        metavar='<directory_path>',
        help='Specify the destination path for the resulting file. Defaults to current directory.',
        type=str,
        default='.'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parser()
    main(args)