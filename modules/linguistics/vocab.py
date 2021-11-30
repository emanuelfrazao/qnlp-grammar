import json

from enum import Enum
from typing import NamedTuple, Union, List, Set, Dict, Literal
from .grammars import ContextFreeRule


class Subject(NamedTuple):
    noun: 'str'
    adjective: 'str'=None
    
class Object(NamedTuple):
    noun: 'str'
    adjective: 'str'=None
    
class HierarchicalSentence(NamedTuple):
    transitive_verb: 'str'
    subject: Subject
    object: Object
    
    

class WordType(Enum):
    NOUN = 'NOUN'
    ADJ = 'ADJ'
    VERB = 'VERB'
    UNK = 'UNK'
    
    def __init__(self, type_):
        self.type_ = type_
        
    def is_noun(self) -> 'bool':
        return self.type_ == 'NOUN'
    
    def is_adjective(self) -> 'bool':
        return self.type_ == 'ADJ'
    
    def is_verb(self) -> 'bool':
        return self.type_ == 'VERB'


class TypedVocabulary:
    
    def __init__(self,
                 nouns: Set[str],
                 adjectives: Set[str],
                 verbs: Set[str],
                 default_noun: 'str'='<unk_noun>',
                 default_adj: 'str'='<unk_adj>',
                 default_verb: 'str'='<unk_verb>',
                 default_unk: 'str'='<unk_word>') -> None:
        
        self.nouns = set(nouns) if not type(nouns) == set else nouns
        self.adjectives = set(adjectives) if not type(adjectives) == set else adjectives
        self.verbs = set(verbs) if not type(verbs) == set else verbs
        self.default_noun = default_noun
        self.default_adj = default_adj
        self.default_verb = default_verb
        self.default_unk = default_unk
        self._defaults = {self.default_noun, self.default_adj, self.default_verb, self.default_unk}
        
    
    def get_word_type(self, word):
        
        if word in self.nouns or word == self.default_noun:
            return WordType('NOUN')
        elif word in self.adjectives or word == self.default_adj:
            return WordType('ADJ')
        elif word in self.verbs or word == self.default_verb:
            return WordType('VERB')
        else:
            return WordType('UNK')
    
    
    def get_default_token(self, word_type: 'Union[str, WordType]') -> 'str':
        
        if isinstance(word_type, str):
            word_type = WordType(word_type)
            
        if word_type.is_noun():
            return self.default_noun
        
        elif word_type.is_adjective():
            return self.default_adj
        
        elif word_type.is_verb():
            return self.default_verb
        
        else:
            return self.default_unk


    def parse_sentence(self, sentence: List['str']) -> HierarchicalSentence:
        
        # Find the transitive verb and its index in the sentence
        verb = next(filter(lambda token: self.get_word_type(token).is_verb(), sentence), None)
        verb_index = sentence.index(verb)
        
        # Grab subject phrase and object phrase from the sentence
        subject_phrase, object_phrase = sentence[:verb_index], sentence[verb_index+1:]
        
        # Get the noun and possible adjective for the subject phrase
        subject_noun = next(filter(lambda token: self.get_word_type(token).is_noun(), subject_phrase), None)
        subject_adj = next(filter(lambda token: self.get_word_type(token).is_adjective(), subject_phrase), None)
        
        # Get the noun and possible adjective for the object phrase
        object_noun = next(filter(lambda token: self.get_word_type(token).is_noun(), object_phrase), None)
        object_adj = next(filter(lambda token: self.get_word_type(token).is_adjective(), object_phrase), None)
        
        return HierarchicalSentence(verb, Subject(subject_noun, subject_adj), Object(object_noun, object_adj))
    

    @classmethod
    def from_json(cls, json_path: 'str'):
        # Get json content
        with open(json_path, 'r') as file:
            content = json.load(file)
            
        return cls(**content)
        
    
    def save_as_json(self, json_path: 'str'):
        
        # Get json content
        content = {attr: list(value) if isinstance(value, set) else value
                    for attr, value in self.__dict__.items() if not attr.startswith('_')}
        
        with open(json_path, 'w') as file:
            json.dump(content, file)        
        
    
    def __contains__(self, word):
        return word in self.nouns or word in self.adjectives or word in self.verbs or word in self._defaults

                
                
                
    

class VocabularyParser:

    types = ['Nobj', 'Nsbj', 'ADJ', 'VB']
    categories = ['FOOD', 'IT', 'common']

    def __init__(self, json_path: 'str') -> None:
        
        # Get json content
        with open(json_path, 'r') as file:
            content = json.load(file)

        # If content is valid, set content attribute
        self.assert_consistent_vocabulary(content)
        self.content = content


    def assert_consistent_vocabulary(self, content: 'Dict[str, Dict[str, List[str]]]') -> 'bool':

        # Assert consistent data
        assert set(content) == set(self.types), \
            f"Vocabulary given does not satisfy required types {self.types} as primary keys."
        
        assert all(set(content.get(type_)) == set(self.categories) for type_ in self.types), \
            f"Vocabulary given does not satisfy required final categories {self.categories} for each type."


    def get_context_free_rules(self,
                              category: 'Literal["FOOD", "IT"]'=None,
                              strictly: 'bool'=False) -> 'List[ContextFreeRule]':
        
        # Get categories
        categories = [category] + (['common'] if not strictly else []) \
            if category \
                else self.categories.copy()
        
        # Filter for given categories
        filtered_vocab = {
            type_: [word for list_ in map(categories_dict.get, categories) for word in list_]
                for type_, categories_dict in self.content.items()
        }

        # Build rules of the form type -> word
        rules = [ContextFreeRule(type_, [word]) for type_, words in filtered_vocab.items() for word in words]

        return rules
    
    
    def get_vocab(self) -> TypedVocabulary:
        
        flattened_content = {type_: [word for list_ in categories_dict.values() for word in list_]
                                for type_, categories_dict in self.content.items()}
            
        return TypedVocabulary(
            nouns = set(flattened_content['Nsbj'] + flattened_content['Nobj']),
            adjectives = set(flattened_content['ADJ']),
            verbs = set(flattened_content['VB'])
        )
        
        
    def _flatten_content(self, categories: List) -> Dict:
        
        flattened_content = {
            type_: [word for list_ in map(categories_dict.get, categories) for word in list_]
                for type_, categories_dict in self.content.items()
        }
        
        return flattened_content
        