"""Script for splitting and saving data into train, validation and test.
"""
## Modules
import argparse
from pathlib import Path
from numpy import random
    
# Costum modules path
import sys
sys.path.insert(0, Path(__file__).parent.parent.as_posix())
from modules.preprocessing import get_data_from_source, train_test_split

def main(args):
    
    # # Assert available paths
    assert Path(args.data).exists(), f"Data path given does not exist."
    assert (destination_path := Path(args.destination)).exists(), f"Destination path given does not exist."
    
    # # Set random seed
    if args.random_seed is not None:
        random.seed(args.random_seed)
    
    # Get data
    data = get_data_from_source(args.data)

    # Split data
    train_data, val_data, test_data = train_test_split(data, train_ratio=.65, test_ratio=.2, shuffle=True)
    
    # Save training data
    with open(destination_path / 'data_train.txt', 'w') as file:
        file_content = '\n'.join([','.join(instance) for instance in train_data])
        file.writelines(file_content)
        
    # Save validation data
    with open(destination_path / 'data_val.txt', 'w') as file:
        file_content = '\n'.join([','.join(instance) for instance in val_data])
        file.writelines(file_content)
        
    # Save testing data
    with open(destination_path / 'data_test.txt', 'w') as file:
        file_content = '\n'.join([','.join(instance) for instance in test_data])
        file.writelines(file_content)
    
    
    

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
    Script for splitting and saving data into train, validation and test.
    
example:
    python .\data_scripts\split.py -d .\data\data.txt -dest .\data\ -r 3''',
    )

    # # Data
    parser.add_argument(
        '-d', '--data',
        metavar='<file_path>.txt',
        help='Specify the data path. Required.',
        type=str,
        required=True
    )
    
    # # Random seed
    parser.add_argument(
        '-r', '--random_seed',
        metavar='<int>',
        help='Specify the random seed to be used. Defaults to None.',
        type=int,
        default=None
    )
    
    # # Destination
    parser.add_argument(
        '-dest', '--destination',
        metavar='<directory_path>',
        help='Specify the destination path for the resulting files. Defaults to current directory.',
        type=str,
        default='.'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parser()
    main(args)
    


    
    