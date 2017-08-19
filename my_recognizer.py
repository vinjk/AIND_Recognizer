import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    # TODO implement the recognizer
    # return probabilities, guesses
    for test_idx in range(0, len(test_set.get_all_Xlengths())):

        score_max = float("-Inf")
        word_match = None
        dict_probs = dict()

        X, lengths = test_set.get_all_Xlengths()[test_idx]

        for train_word, model in models.items():
            try:
                dict_probs[train_word] = model.score(X,lengths)
                if score_max < dict_probs[train_word]:
                    score_max = dict_probs[train_word]
                    word_match = train_word
            except:
                dict_probs[train_word] = float("-Inf")

            # dict_probs[train_word] = logL
        probabilities.append(dict_probs)
        guesses.append(word_match)

    return probabilities, guesses