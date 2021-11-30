# Argument processing modules
from pathlib import Path
import argparse

# Typing modules
from typing import NamedTuple, List, Tuple, Dict, Literal

# Costum modules path
import sys
sys.path.insert(0, Path(__file__).parent.parent.as_posix())

# Costum modules
from modules.linguistics.vocab import VocabularyParser
from modules.linguistics.grammars import ContextFreeRule, ContextFreeGrammar


class DataInstance(NamedTuple):
    """Named tuple for representing a single instance in the dataset.
    
    Args:
        first_sentence (str): The first sentence of the instance.
        second_sentence (str): The second sentence of the instance.
        label (int): The integer boolean representing whether the instances belong to the same category.
        
    """
    first_sentence: 'str'
    second_sentence: 'str'
    label: 'str'


class DataBuilder:
    """Class for generating instances of the form (sentence_1, sentence_2, label)
    from a given vocabulary with categories {'IT', 'FOOD'} and word types {VB, ADJ, Nobj, Nsbj}
    with a predefined context free grammar.
    """    

    # Set predefined rules for the grammars
    grammar_base_rules = [
        ContextFreeRule('S', ['NsbjP', 'VP']),
        ContextFreeRule('NsbjP', ['ADJ', 'Nsbj']),
        ContextFreeRule('NsbjP', ['Nsbj']),
        ContextFreeRule('VP', ['VB', 'NobjP']),
        ContextFreeRule('NobjP', ['ADJ', 'Nobj']),
        ContextFreeRule('NobjP', ['Nobj']),
    ]

    def __init__(self: int,
                 vocab_file: 'str',
                 random_seed: 'int'=None) -> None:
        """Sets up the vocabulary and the grammars for each category.

        Args:
            vocab_file (str): Path to the vocabulary json file.
            random_seed (int, optional): Random seed to be fed into the grammars. Defaults to None.
        """        
        ## Set vocabulary parser
        self.vocab_parser = VocabularyParser(vocab_file)

        ## Build rules for grammars out of the vocabulary
        # FOOD rules
        vocab_food_rules = self.vocab_parser.get_context_free_rules('FOOD', strictly=False)
        self.rules_food = self.grammar_base_rules + vocab_food_rules
    
        # IT rules
        vocab_it_rules = self.vocab_parser.get_context_free_rules('IT', strictly=False)
        self.rules_it = self.grammar_base_rules + vocab_it_rules

        ## Set IT and FOOD grammars
        self.grammar_food = ContextFreeGrammar(self.rules_food, random_seed=random_seed)
        self.grammar_it = ContextFreeGrammar(self.rules_it, random_seed=random_seed)


    def generate_instances(self,
                           size: 'int') -> 'List[DataInstance]':
        """Generates specified number of random data instances equally distributed between labels.

        Args:
            size (int): Number of instances to be generated.
        Returns:
            List[DataInstance]: List of generated instances.
        """        
        instances = []

        # Get number of instances of each type
        n_different_categories = size // 2 + size % 2
        n_same_categories = size - n_different_categories

        ## Build instances of each type
        # From different categories (with label 0)
        for i in range(n_different_categories):
            if i % 2 == 0:
                instances.append(self.generate_single_instance('FOOD', 'IT'))
            else:
                instances.append(self.generate_single_instance('IT', 'FOOD'))

        # Both from IT category (with label 1)
        for _ in range(n_same_categories // 2 + n_same_categories % 2):
            instances.append(self.generate_single_instance('IT', 'IT'))

        # Both from FOOD category (with label 1)
        for _ in range(n_same_categories // 2):
            instances.append(self.generate_single_instance('FOOD', 'FOOD'))

        return instances


    def generate_single_instance(self,
                                 first: 'Literal["FOOD", "IT"]',
                                 second: 'Literal["FOOD", "IT"]') -> 'DataInstance':
        
        """Generates a single random data instance from the given vocabulary
        with two sentences of specified categories and the correspondent label for whether the categories are the same.
        
        Args:
            first (Literal["FOOD", "IT"]): Category of first sentence.
            second (Literal["FOOD", "IT"]): Category of second sentence.
        
        Returns:
            DataInstance: The generated data instance.
        """        
        # Assert validity
        assert first in ['FOOD', 'IT'] and second in ['FOOD', 'IT'], f"Invalid input. Inputs must belong to the category 'FOOD' or 'IT'."

        # Set pointers to the grammars being used in each sentence
        first_sentence_grammar = self.grammar_it if first == 'IT' else self.grammar_food
        second_sentence_grammar = self.grammar_it if second == 'IT' else self.grammar_food

        # Return sentences and respective label
        instance = DataInstance(first_sentence_grammar.generate_random_sentence(), second_sentence_grammar.generate_random_sentence(), int(first == second))
        return instance




def main(args: 'argparse.Namespace') -> None:
    """Builds the data - conditioned on the arguments given - and saves it in the specified directory.
    
    Args:
        args (argparse.Namespace): The object with stored arguments.
    """    
    # Retrieve command line arguments and verify consistency
    assert (destination_path := Path(args.destination)).exists(), f"Destination path {destination_path.as_posix()!r} doesn't exist."
    
    # Instantiate builder class
    builder = DataBuilder(args.vocabulary, args.random_seed)

    # Generate given size of instances
    data = builder.generate_instances(args.size)

    # Save data to the given directory
    with open(destination_path / 'data.txt', 'w') as file:
        file_content = '\n'.join([','.join(map(str, instance)) for instance in data])
        file.writelines(file_content)


    
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
    oi''',
    )

    # Vocabulary
    parser.add_argument(
        '-v', '--vocabulary',
        metavar='<file_path>.json',
        help='Specify the number of instances to be generated. Required.',
        type=str,
        required=True
    )

    # Size
    parser.add_argument(
        '-s', '--size',
        metavar='int',
        help='Specify the number of instances to be generated. Required.',
        type=int,
        required=True
    )

    # Random seed
    parser.add_argument(
        '-r', '--random_seed',
        metavar='<int>',
        help='Specify the random seed to be used. Defaults to None.',
        type=int,
        default=None
    )

    # Destination
    parser.add_argument(
        '-d', '--destination',
        metavar='<directory_path>',
        help='Specify the destination path for the resulting file. Defaults to current directory.',
        type=str,
        default='.'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parser()
    main(args)

