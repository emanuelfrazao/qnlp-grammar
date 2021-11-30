from typing import NamedTuple, List
import random


class ContextFreeRule(NamedTuple):
    """Named tuple for representing a context free grammar' rule.
    
    Args:
        left (str): The left side of the rule.
        right (str): The right side of the rule.
        
    """
    left: 'str'
    right: 'str'

class ContextFreeGrammar:
    """Class
    """    

    def __init__(self,
                 rules: 'List[ContextFreeRule]',
                 start_symbol: 'str'='S',
                 random_seed: 'int'=None) -> None:

        # Set random seed
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)

        self.rules = rules
        self.start_symbol = start_symbol
        
        # Get nonterminal symbols and assert empty sentence' symbol 'S' is there
        self.nonterminal_symbols = set(rule.left for rule in rules)
        assert start_symbol in self.nonterminal_symbols, \
            f"Symbol {start_symbol} (for an empty sentence) should be a nonterminal symbol present in the grammar rules."
        
        # Get all symbols
        self.symbols = self.nonterminal_symbols.union(symbol for rule in rules for symbol in rule.right)
        
        # Get terminal symbols
        self.terminal_symbols = self.symbols.difference(self.nonterminal_symbols)
        
        

    def generate_random_sentence(self,
                                 depth: 'int'=100) -> 'str':
        
        # Initialize empty sentence
        sentence = [self.start_symbol]

        # Iteratively apply random rule until the sentence is terminal
        for _ in range(depth):
            if not self.is_terminal(sentence):
                self._apply_random_rule_inplace(sentence)
            else:
                break
        
        return ' '.join(sentence)
        
    def _apply_random_rule_inplace(self,
                                   sentence: 'List[str]') -> None:
        
        # Randomly choose a nonterminal symbol in the sentence (given the assertion that it exists)
        possible_words_indices = [i for i, symbol in enumerate(sentence) if symbol in self.nonterminal_symbols]
        word_index = random.choice(possible_words_indices)
        
        # Randomly choose rule with that left symbol  
        possible_rules_indices = [i for i, rule in enumerate(self.rules) if rule.left == sentence[word_index]]
        rule_index = random.choice(possible_rules_indices)
        
        self._apply_rule_inplace(sentence, word_index, self.rules[rule_index])
        
        
    def _apply_rule_inplace(self,
                            sentence: 'List[str]',
                            index: 'int',
                            rule: 'ContextFreeRule') -> None:
        
        assert sentence[index] == rule.left
        
        sentence[index:index+1] = rule.right
        
    
    def is_terminal(self,
                    sentence: 'List[str]') -> 'bool':
        
        return not any(symbol in self.nonterminal_symbols for symbol in sentence)