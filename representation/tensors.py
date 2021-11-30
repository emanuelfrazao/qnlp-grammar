from typing import Union, Literal, List, Dict, Tuple
from collections import Counter
import numpy as np

from ..linguistics.relations import Context, NounRelations, AdjectiveRelations, VerbRelations, ContextWord
from ..linguistics.vocab import HierarchicalSentence, TypedVocabulary
from .utils import get_context_words_distribution


class BaseTensorizer:
    """Base class for a tensorizer with all the functionality for initializing its subclasses.
    """
    def __init__(self,
                 corpus: 'List[str]',
                 reference_vocab: 'TypedVocabulary',
                 cutoff: 'int'=1) -> None:
        """Builds the vocabulary based on the corpus, a reference typed vocabulary, and the minimum number of words (cutoff).
        Then, it grabs all grammatical relations between words in the corpus - used for tensorizing them in a more efficient way.

        Args:
            corpus (List[str]): The list of sentences to be considered in order to train the tensor representations.
            reference_vocab (TypedVocabulary): The TypedVocabulary object that serves as a reference table for each word's type.
            cutoff (int, optional): The minimum number of occurences in the corpus for a word to be in the vocabulary. Defaults to 1.
        """        
                
        # Save parameters
        self.source_corpus = corpus
        self.reference_vocab = reference_vocab
        self.cutoff = cutoff if cutoff > 0 else 1
        
        # Set corpus vocabulary based on occuring words in the corpus and given mininum number of occurences
        self.vocab = self._build_corpus_vocabulary()
    
        # Get tokenize corpus
        self.tokenized_corpus = self._build_tokenized_corpus()  
        
        # Get words relations instances from the tokenized corpus
        words_relations = self._grab_words_relations()
        self.nouns_relations = words_relations['nouns']
        self.adjectives_relations = words_relations['adjectives']
        self.verbs_relations = words_relations['verbs']
        
        self.is_trained = False
        # # Initialize tensor attributes
        # self.context_base = None
        # self.space_dimension = None
        # self.nouns2tensors = None
        # self.adjectives2tensors = None
        # self.verbs2tensors = None
   
        
    
    def build_tensors(self):
        """Builds inplace the tensors for nouns, adjectives and verbs.

        Raises:
            NotImplementedError: To be implemented in a subclass.
        """
        raise NotImplementedError
    
    def tensorize(self, 
                  sentence: 'str') -> np.ndarray:
        
        assert self.is_trained, "Tensorizer is not trained yet. Call 'build_tensors' method with a context base in order to train the representations."
        
        # Get a hierarchical representation of the sentence
        t_sentence = self.tokenize(sentence)
        h_sentence = self.vocab.parse_sentence(t_sentence)
        
        # Get tensors for subject phrase and noun phrase
        sbjNP_tensor = np.matmul(
            self.adjectives2tensors[h_sentence.subject.adjective or self.vocab.default_adj],
            self.nouns2tensors[h_sentence.subject.noun]
        )
        
        objNP_tensor = np.matmul(
            self.adjectives2tensors[h_sentence.object.adjective or self.vocab.default_adj],
            self.nouns2tensors[h_sentence.object.noun]
        )
        
        # Get corresponding verb tensors
        verb_sbj_tensor, verb_obj_tensor = self.verbs2tensors[h_sentence.transitive_verb]
        
        # Perform the operation and return
        return np.matmul(verb_sbj_tensor, sbjNP_tensor) + np.matmul(verb_obj_tensor, objNP_tensor)
    
    
    ######################################### Auxiliary methods for instantiating the class #######################
    
    def _build_corpus_vocabulary(self) -> 'TypedVocabulary':
        """Builds a typed vocabulary based on the corpus, the reference vocabylary and cutoff.

        Returns:
            TypedVocabulary: The typed vocabulary built.
        """        
        
        # Get most present words in corpus
        most_present_words = [word for word, count in self.get_corpus_word_counts().items() if count >= self.cutoff]
        
        # Return filtered vocabulary of the most present words in corpus that are typed in the reference vocab
        return TypedVocabulary(
            nouns=self.reference_vocab.nouns.intersection(most_present_words),
            adjectives=self.reference_vocab.adjectives.intersection(most_present_words),
            verbs=self.reference_vocab.verbs.intersection(most_present_words),
            default_noun=self.reference_vocab.default_noun,
            default_adj=self.reference_vocab.default_adj,
            default_verb=self.reference_vocab.default_verb,
            default_unk=self.reference_vocab.default_unk
        )
        
        
    def _build_tokenized_corpus(self) -> 'List[List[str]]':
        """Replaces each sentence in the corpus by its tokenized form.

        Returns:
            List[List[str]]: The list of all tokenized sentences, where each one is a list of tokens.
        """        
        return [self.tokenize(sentence) for sentence in self.source_corpus]
                
    
    def _grab_words_relations(self) -> 'Dict[str, Dict[str, List]]':
        """Grabs all the grammatical relations between words, for every word, and organized by type of word (noun, adjective, verb)
        Grammatical relations can either be NounRelations, AdjectiveRelations or VerbRelations.
        
        Returns:
            Dict[str, Dict[str, List]]: A dictionary with keys 'nouns', 'adjectives' and 'verbs'. 
                                        Each value is again a dictionary with the words of the respective type as keys,
                                        and their grammatical relations as values.                 
        """        
        # Initiate grammatical relations dictionary for every noun, adjective and verb (including respective default values)
        words_relations = {
            'nouns': {noun: [] for noun in self.vocab.nouns.union({self.vocab.default_noun})},
            'adjectives': {adjective: [] for adjective in self.vocab.adjectives.union({self.vocab.default_adj})},
            'verbs': {verb: [] for verb in self.vocab.verbs.union({self.vocab.default_verb})},
        }
        
        # Save, for each word in the corpus (given its sentence boundary), its grammatical relations conditioned on its type
        for t_sentence in self.tokenized_corpus:
            # Parse sentence into a hierarchical sentence (with structure)
            h_sentence: HierarchicalSentence = self.vocab.parse_sentence(t_sentence)
            
            # Grab and save its grammatical relations
            self._append_sentence_relations(words_relations, h_sentence)
            
        return words_relations
    
    
    @staticmethod
    def _append_sentence_relations(words_relations: 'Dict[str, Dict[str, List]]',
                                   h_sentence: 'HierarchicalSentence') -> None:
        """A utility method for '_grab_words_relations' that appends the grammatical relations found in a hierarchical sentence to its 'words_relations' object.

        Args:
            words_relations (Dict[str, Dict[str, List]]): The grammatical relations already registered.
            h_sentence (HierarchicalSentence): The hierarchical sentence from which to grab new grammatical relations.
        """               
        # Save transitive verb relations for verb in the sentence
        words_relations['verbs'][h_sentence.transitive_verb].append(VerbRelations.from_sentence(h_sentence))
        
        # Save nouns relations, for subject and object in a sentence
        words_relations['nouns'][h_sentence.subject.noun].append(NounRelations.from_sentence(h_sentence, 'sbj'))
        words_relations['nouns'][h_sentence.object.noun].append(NounRelations.from_sentence(h_sentence, 'obj'))
        
        # Save adjective relations, if present, for subject and object
        if (sbj_adjective := h_sentence.subject.adjective) is not None:
            words_relations['adjectives'][sbj_adjective].append(AdjectiveRelations.from_sentence(h_sentence, 'sbj'))
        
        if (obj_adjective := h_sentence.object.adjective) is not None:
            words_relations['adjectives'][obj_adjective].append(AdjectiveRelations.from_sentence(h_sentence, 'obj'))
    
   
    ######################################### Auxiliary methods to be used outside ################################

    def get_corpus_word_counts(self) -> 'Dict[str, int]':
        """Retrieves the counts of each word in the corpus.

        Returns:
            Dict[str, int]: The words counter.
        """
        sequence_all_words = [word for sentence in self.source_corpus for word in sentence.split()]
        
        return Counter(sequence_all_words)
    
    
    def tokenize(self,
                 sentence: 'str') -> 'List[str]':
        """Turns a sentence given as a lower case string into a list of tokens.

        Args:
            str: The sentence to consider.
            
        Returns:
            List[str]: The list of tokens corresponding to the sentence.
        """        
        return [self.get_token(word) for word in sentence.split()]
    
    
    def get_token(self,
                  word: 'str') -> 'str':
        """Retrieves the token of a given word.
        If it doesn't exist in the reference vocabulary (it's type is unkown), it returns the default unkown token.
        If it does, but it doesn't exist in the corpus vocabulary, returns the default token corresponding to the type of the word.
    
        Arg:
            word (str): The word considered.
        Returns:
            str: The corresponding token of the word.
        """        
        if word not in self.reference_vocab:
            return self.reference_vocab.default_unk
        elif word in self.vocab:
            return word
        else:
            word_type = self.reference_vocab.get_word_type(word)
            return self.vocab.get_default_token(word_type)
    











class ExtensionalTensorizer(BaseTensorizer):
    """Class for creating vector space representations of words (nouns, adjectives and verbs) in a corpus, based on different type-logic syntactic roles.
    """
    def __init__(self,
                 corpus: 'List[str]',
                 reference_vocab: 'TypedVocabulary',
                 cutoff: 'int'=1) -> None:
        """Builds the vocabulary based on the corpus, a reference typed vocabulary, and the minimum number of words (cutoff).
        Then, it grabs all grammatical relations between words in the corpus - used for tensorizing them in a more efficient way.

        Args:
            corpus (List[str]): The list of sentences to be considered in order to train the tensor representations.
            reference_vocab (TypedVocabulary): The TypedVocabulary object that serves as a reference table for each word's type.
            cutoff (int, optional): The minimum number of occurences in the corpus for a word to be in the vocabulary. Defaults to 1.
        """
        super().__init__(corpus, reference_vocab, cutoff)
        self.is_trained = False
    
    ######################################### Main method for tensorizing #########################################
    
    def build_tensors(self,
                      context_base: 'Context') -> 'Tuple[Context, Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]':
        """The main method for tensorizing the words in the corpus' vocabulary given a context base.

        Args:
            context_base (Context): The context base.
            
        Returns:
            Tuple[Context, Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
                Context: The context base built or received.
                Dict[str, np.ndarray]: The dictionary with nouns as keys and the corresponding noun tensor representation as values.
                Dict[str, np.ndarray]: The dictionary with adjectives as keys and the corresponding adjective tensor representation as values.
                Dict[str, np.ndarray]: The dictionary with verbs as keys and the corresponding verb tensor representation as values.
        """
        # Get space dimension
        self.context_base = context_base
        self.space_dimension = len(context_base)
        
        # Build tensors for nouns, adjectives and verbs
        self._build_noun_tensors()
        self._build_adjective_tensors()
        self._build_verb_tensors()
        
        self.is_trained = True


    ######################################### Auxiliary methods for tensorizing ###################################
    
    
    def _build_noun_tensors(self) -> None:
        """Builds inplace the tensor representation of nouns given the context base.
        """
        # Initialize
        self.nouns2tensors = {noun: np.zeros(self.space_dimension) for noun in self.vocab.nouns}
        
        # Go through each noun in the vocabulary
        for noun, noun_relations_instances in self.nouns_relations.items():
            if noun != self.vocab.default_noun:
                # Build its tensor representation based on its instances in the vocabulary
                for relations_instance in noun_relations_instances:
                    for context_word in relations_instance.as_context_words():
                        if (base_index := self.context_base.index(context_word)) is not None:
                            self.nouns2tensors[noun][base_index] += 1

        # Add default noun representation        
        self.nouns2tensors[self.vocab.default_noun] = np.ones(self.space_dimension)


    def _build_adjective_tensors(self) -> None:
        """Builds inplace the tensor representation of adjectives given the nouns representations.
        """  
        # Initialize
        self.adjectives2tensors = {adj: np.zeros(self.space_dimension) for adj in self.vocab.adjectives}
        
        # Go through each adjective in the vocabulary
        for adj, adj_relations_instances in self.adjectives_relations.items():
            if adj != self.vocab.default_adj:
                
                # Build the representation of the adjective given its instances in the corpus
                for relations_instance in adj_relations_instances:
                    self.adjectives2tensors[adj] += self.nouns2tensors[relations_instance.arg]
                self.adjectives2tensors[adj] = np.diag(self.adjectives2tensors[adj])
        
        # Add the default adjective representation 
        self.adjectives2tensors[self.vocab.default_adj] = np.eye(self.space_dimension)
            
                
    def _build_verb_tensors(self) -> None:
        """Builds inplace the tensor representation of verbs given the nouns representations.
        """  
        # Initialize
        self.verbs2tensors = {verb: np.zeros((self.space_dimension, self.space_dimension)) for verb in self.vocab.verbs}

        # Go through each verb in the vocabulary
        for verb, verb_relations_instances in self.verbs_relations.items():
            if verb != self.vocab.default_verb:
                # Build the representation of the adjective given its instances in the corpus
                for relations_instance in verb_relations_instances:
                    self.verbs2tensors[verb] += np.outer(self.nouns2tensors[relations_instance.sbj], self.nouns2tensors[relations_instance.obj])
                self.verbs2tensors[verb] = (self.verbs2tensors[verb].transpose(), self.verbs2tensors[verb])
            
        # Add default verb representation
        self.verbs2tensors[self.vocab.default_verb] = (np.eye(self.space_dimension), np.eye(self.space_dimension))
    

    
    
    

