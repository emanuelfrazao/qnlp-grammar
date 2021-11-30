"""Script for running the simple bilinear model on training and validation data. 
May be used as a module by providing a Namespace to the main method.
"""
## Modules
import argparse
from pathlib import Path
import torch

# Costum modules path
import sys
sys.path.insert(0, Path(__file__).parent.parent.as_posix())

from modules.preprocessing import preprocess, get_data_from_source, get_sentences, train_test_split
from modules.linguistics.vocab import TypedVocabulary
from modules.linguistics.relations import Context
from modules.representation.tensors import ExtensionalTensorizer
from modules.modelling.models.neuralnetworks import SimpleBilinearModel
from modules.modelling.trainer import NNTrainer



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
    
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()
    X_val = torch.from_numpy(X_val).float()
    y_val = torch.from_numpy(y_val).float()
    
    # # Train model
    model = SimpleBilinearModel(len(context_base), args.dropout)
    optim = torch.optim.Adam(model.parameters(), args.lr)
    loss = torch.nn.BCELoss()
    acc = lambda true, pred: (true == pred).sum().item() / len(true)
    trainer = NNTrainer(model, optim, loss, acc, 3)
    
    trainer.fit(X_train, y_train, X_val, y_val, args.epochs, args.eval_every, args.batch, args.verbose)
    
    trainer._save_plot_fit_history(args.destination)
    



def arg_parser() -> 'argparse.Namespace':
    """Parses command line arguments.
    
    Returns:
        argparse.Namespace: The object with stored arguments.
    """    
    ## Initializer
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=
        '''
description:
    Script for running the simple bilinear model on training and validation data. Saves the fitting history in the destionation path provided.
    
example:
    python .\model_scripts\simple_bilinear_training.py 
        --train_data .\data\data_train.txt 
        --val_data .\data\data_val.txt 
        -v .\\vocabs\\reference_vocab.json 
        -c .\contexts\context.json 
        --dropout .8 
        --lr .01 
        -dest .''',
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
    
    # # Tensorizer arguments
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
    
    # Context
    parser.add_argument(
        '-c', '--context',
        metavar='<file_path>.json',
        help='Specify the context basis path. Required.',
        type=str,
        required=True
    )
    
    # # Trainer arguments
    # Learning rate
    parser.add_argument(
        '--lr',
        metavar='<float>',
        help='Specify the learning rate of the optimizer. Defaults to 0.03.',
        type=float,
        default=0.03
    )
    
    # Epochs
    parser.add_argument(
        '--epochs',
        metavar='<int>',
        help='Specify the number of epochs for fitting. Defaults to 30.',
        type=int,
        default=30
    )
    
    # Evaluation period
    parser.add_argument(
        '--eval_every',
        metavar='<int>',
        help='Specify the interval of epochs for printing the results. Defaults to 2.',
        type=int,
        default=2
    )
    
    # Batch size
    parser.add_argument(
        '--batch',
        metavar='<int>',
        help='Specify the number of batches for fitting. Defaults to 8.',
        type=int,
        default=8
    )
    
    # Verbose
    parser.add_argument(
        '--verbose',
        metavar='<bool>',
        help='Specify whether or not to print iterations. Defaults to True.',
        type=bool,
        default=True
    )
    
    # # Network arguments
    # Dropout probability
    parser.add_argument(
        '--dropout',
        metavar='float',
        help='Specify the probability of dropout for the bilinear layer. Defaults to 0.8.',
        type=float,
        default=0.8
    )

    # # Destination
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