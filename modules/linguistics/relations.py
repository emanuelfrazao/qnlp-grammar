from typing import NamedTuple, Literal, List, Optional
from collections import Hashable
import json

from .vocab import HierarchicalSentence


class NounRelations(NamedTuple):
    """Named tuple representing the grammatical relations of a noun in a boundary sentence.
    
    Args:
        adj_arg (Optional[str]): The adjective of which the noun is argument to. Defaults to None for when there is none.
        verb_sbj (Optional[str]): The verb of which the noun is subject to. Defaults to None for when there is none.
        verb_obj (Optional[str]): The verb of which the noun is object to. Defaults to None for when there is none.
    
    """
    adj_arg: 'str' = None
    verb_sbj: 'str' = None
    verb_obj: 'str' = None
    
    @classmethod
    def from_sentence(cls, sentence: HierarchicalSentence, noun_phrase_type: 'Literal["sbj", "obj"]'):
        if noun_phrase_type == 'sbj':
            return cls(adj_arg=sentence.object.adjective, verb_sbj=sentence.transitive_verb)
        elif noun_phrase_type == 'obj':
            return cls(adj_arg=sentence.subject.adjective, verb_obj=sentence.transitive_verb)
    
    
    def as_context_words(self) -> 'List[ContextWord]':
        
        context_relations = []
        if self.adj_arg is not None:
            context_relations.append(ContextWord(self.adj_arg, 'arg'))
        if self.verb_sbj is not None:
            context_relations.append(ContextWord(self.verb_sbj, 'sbj'))
        if self.verb_obj is not None:
            context_relations.append(ContextWord(self.verb_obj, 'obj'))
            
        return context_relations
        

class AdjectiveRelations(NamedTuple):
    """Named tuple representing the grammatical relations of an adjective in a boundary noun phrase.
    
    Args:
        arg (str): The noun to which the adjective refers.
        
    """
    arg: 'str'
    
    @classmethod
    def from_sentence(cls, sentence: HierarchicalSentence, noun_phrase_type: 'Literal["sbj", "obj"]'):
        if noun_phrase_type == 'sbj':
            return cls(arg=sentence.subject.noun)
        elif noun_phrase_type == 'obj':
            return cls(arg=sentence.object.noun)

    
class VerbRelations(NamedTuple):
    """Named tuple representing the grammatical relations of a transitive verb in a boundary sentence.

    Args:
        sbj: The noun subject of the verb.
        obj: The noun object of the verb.
        
    """
    sbj: 'str'
    obj: 'str'
    
    @classmethod
    def from_sentence(cls, sentence: HierarchicalSentence):
        return cls(sentence.subject.noun, sentence.object.noun)

    
class ContextWord(NamedTuple):
    """Named tuple representing a word in the context set (i.e., a base vector of the nouns vector space).
    """
    word: 'str'
    hole: 'Literal["arg", "sbj", "obj"]'
    


class Context:

    def __init__(self,
                 context_words: 'List[ContextWord]') -> None:
        
        self.index2word = dict(enumerate(context_words))
        self.word2index = {word: i for i, word in self.index2word.items()}
        
    
    def index(self,
              word: ContextWord) -> 'Optional[int]':
        
        return self.word2index.get(word, None)
    
    @classmethod
    def from_lists(cls,
                   adjectives_list: 'List[str]',
                   verb_sbj_list: 'List[str]',
                   verb_obj_list: 'List[str]') -> 'Context':
    
        context_words = [ContextWord(adjective, 'arg') for adjective in adjectives_list] \
                        + [ContextWord(verb, 'sbj') for verb in verb_sbj_list] \
                        + [ContextWord(verb, 'obj') for verb in verb_obj_list]
                        
        return cls(context_words)

    @classmethod
    def from_json(cls,
                  json_path: 'str') -> 'Context':
        """Builds a Context from a json file of the form {index: [word, hole]}.
        
        Example of json content:
            {
                "0": ["great", "arg"],
                "1": ["builds", "sbj"],
                "2": ["evaluates", "obj"]
            }
        """        
        # Get json content
        with open(json_path, 'r') as file:
            content = json.load(file)
        
        context_list = [ContextWord(*value) for key, value in sorted(content.items(), key=lambda item: int(item[0]))]
        return cls(context_list)
        


    def __contains__(self,
                     context_word: ContextWord) -> 'bool':
        return context_word in self.word2index
            
    def __len__(self) -> 'int':
        return len(self.word2index)