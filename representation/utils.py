from typing import Tuple

def get_context_words_distribution(space_dimension,
                              fraction_adjectives) -> Tuple[int, int, int]:
    
    # Get number of adjectives with argument hole
    n_adjectives = int(space_dimension * fraction_adjectives)
    
    # Get number of verbs with subject hole and number of verbs with object hole
    n_verbs = space_dimension - n_adjectives
    n_verbs_sbj = n_verbs // 2
    n_verbs_obj = n_verbs - n_verbs_sbj
    
    return n_adjectives, n_verbs_sbj, n_verbs_obj