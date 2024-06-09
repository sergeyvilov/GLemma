import re
import json
import os
import pickle

import pandas as pd
import numpy as np
from pathlib import Path

def get_verb_lemma_fwdsearch(word, query_verb_dict_fnc, use_longest_subword=True):

    '''
    Get all possible verb lemmas using forward search, prefix-agnostic
    '''

    lemmas = []
    
    for start_idx in range(0,len(word)-2):
        #remove letters one by one, until the rest of the word matches one in the vocabulary
        trial_word = word[start_idx:]
        base_lemma = query_verb_dict_fnc(trial_word)
        if start_idx>2 and word[start_idx-2:start_idx] == 'zu' and base_lemma == trial_word:
            #suspect a zu-infinitive: some prefix on the right + zu + infinitive
            #e.g. abzuheben, aufzuatmen
            lemmas.append(word[0:start_idx-2] + base_lemma) #add without "zu"
        elif base_lemma and not base_lemma.startswith('zu'):
            lemmas.append(word[0:start_idx] + base_lemma)

    if lemmas and (use_longest_subword or len(set(lemmas))==1):
        #all possible splits lead to the same lemma or taking the longest subword allowed
        return lemmas[0]
    else:
        return None

article_constraints = {'dieser': (('f', 'Genitiv Singular'), ('n', 'Genitiv Plural'), ('m', 'Genitiv Plural'), ('m', 'Nominativ Singular'), ('f', 'Dativ Singular'), ('f', 'Genitiv Plural'), ('only_plural', 'Genitiv Plural')), 
 'der': (('f', 'Genitiv Singular'), ('n', 'Genitiv Plural'), ('m', 'Genitiv Plural'), ('m', 'Nominativ Singular'), ('f', 'Dativ Singular'), ('f', 'Genitiv Plural'), ('only_plural', 'Genitiv Plural')), 
 'kein': (('m', 'Nominativ Singular'), ('n', 'Nominativ Singular'), ('n', 'Akkusativ Singular')), 
 'dieses': (('n', 'Nominativ Singular'), ('n', 'Akkusativ Singular'), ('m', 'Genitiv Singular'), ('n', 'Genitiv Singular')), 
 'des': (('m', 'Genitiv Singular'), ('n', 'Genitiv Singular')), 'keines': (('m', 'Genitiv Singular'), ('n', 'Genitiv Singular')), 
 'diesem': (('n', 'Dativ Singular'), ('m', 'Dativ Singular')), 'dem': (('n', 'Dativ Singular'), ('m', 'Dativ Singular')), 
 'keinem': (('n', 'Dativ Singular'), ('m', 'Dativ Singular')), 'diesen': (('n', 'Dativ Plural'), ('only_plural', 'Dativ Plural'), ('f', 'Dativ Plural'), ('m', 'Akkusativ Singular'), ('m', 'Dativ Plural')), 
 'den': (('n', 'Dativ Plural'), ('only_plural', 'Dativ Plural'), ('f', 'Dativ Plural'), ('m', 'Akkusativ Singular'), ('m', 'Dativ Plural')), 
 'keinen': (('n', 'Dativ Plural'), ('only_plural', 'Dativ Plural'), ('f', 'Dativ Plural'), ('m', 'Akkusativ Singular'), ('m', 'Dativ Plural')), 
 'die': (('only_plural', 'Nominativ Plural'), ('n', 'Akkusativ Plural'), ('f', 'Akkusativ Singular'), ('only_plural', 'Akkusativ Plural'), ('f', 'Nominativ Singular'), ('m', 'Nominativ Plural'), ('m', 'Akkusativ Plural'), ('f', 'Nominativ Plural'), ('n', 'Nominativ Plural'), ('f', 'Akkusativ Plural')), 
 'diese': (('only_plural', 'Nominativ Plural'), ('n', 'Akkusativ Plural'), ('f', 'Akkusativ Singular'), ('only_plural', 'Akkusativ Plural'), ('f', 'Nominativ Singular'), ('m', 'Nominativ Plural'), ('m', 'Akkusativ Plural'), ('f', 'Nominativ Plural'), ('n', 'Nominativ Plural'), ('f', 'Akkusativ Plural')), 
 'keine': (('only_plural', 'Nominativ Plural'), ('n', 'Akkusativ Plural'), ('f', 'Akkusativ Singular'), ('only_plural', 'Akkusativ Plural'), ('f', 'Nominativ Singular'), ('m', 'Nominativ Plural'), ('m', 'Akkusativ Plural'), ('f', 'Nominativ Plural'), ('n', 'Nominativ Plural'), ('f', 'Akkusativ Plural')), 
 'keiner': (('f', 'Genitiv Singular'), ('f', 'Genitiv Plural'), ('m', 'Genitiv Plural'), ('f', 'Dativ Singular'), ('n', 'Genitiv Plural'), ('only_plural', 'Genitiv Plural')), 'das': (('n', 'Nominativ Singular'), ('n', 'Akkusativ Singular'))
}

prep_constraints = {
                 'zu': (('m','Dativ Singular'),('n','Dativ Singular'),('f','Dativ Singular'),('m','Dativ Plural'),('n','Dativ Plural'),('f','Dativ Plural')),
                 'von': (('m','Dativ Singular'),('n','Dativ Singular'),('f','Dativ Singular'),('m','Dativ Plural'),('n','Dativ Plural'),('f','Dativ Plural')),
                 'bei': (('m','Dativ Singular'),('n','Dativ Singular'),('f','Dativ Singular'),('m','Dativ Plural'),('n','Dativ Plural'),('f','Dativ Plural')),
                 'durch': (('m','Akkusativ Singular'),('n','Akkusativ Singular'),('f','Akkusativ Singular'),('m','Akkusativ Plural'),('n','Akkusativ Plural'),('f','Akkusativ Plural')),
                 'für': (('m','Akkusativ Singular'),('n','Akkusativ Singular'),('f','Akkusativ Singular'),('m','Akkusativ Plural'),('n','Akkusativ Plural'),('f','Akkusativ Plural')),
                 'um': (('m','Akkusativ Singular'),('n','Akkusativ Singular'),('f','Akkusativ Singular'),('m','Akkusativ Plural'),('n','Akkusativ Plural'),('f','Akkusativ Plural')),
                 'im':(('m','Dativ Singular'),('n','Dativ Singular')),
                 'beim':(('m','Dativ Singular'),('n','Dativ Singular')),
                 'zum':(('m','Dativ Singular'),('n','Dativ Singular')),
                 'vom':(('m','Dativ Singular'),('n','Dativ Singular')),
                 'zur':(('f','Dativ Singular'),),
                 'hintern':(('m','Akkusativ Singular'),),
                 'übern':(('m','Akkusativ Singular'),),
                 'untern':(('m','Akkusativ Singular'),),
                 'ins':(('n','Akkusativ Singular'),),
                 'aufs':(('n','Akkusativ Singular'),),
                 'durchs':(('n','Akkusativ Singular'),),
                 'fürs':(('n','Akkusativ Singular'),),
                 'ums':(('n','Akkusativ Singular'),),
                 'vors':(('n','Akkusativ Singular'),),
                 'übers':(('n','Akkusativ Singular'),),
                 'unters':(('n','Akkusativ Singular'),),
                 'hinterm':(('m','Dativ Singular'),('n','Dativ Singular')),
                 'überm':(('m','Dativ Singular'),('n','Dativ Singular')),
                 'unterm':(('m','Dativ Singular'),('n','Dativ Singular')),
                 'vorm':(('m','Dativ Singular'),('n','Dativ Singular')),
                }

def get_base_determiner(word):
    '''
    Check if a given word is a determiner and return the base form
    '''
    kein_style = re.match(r'(mein|dein|sein|ihr|Ihr|euer|unser|ein|kein|welch|solch|manch)($|e[rsnm]?$)',word)
    if kein_style:
        return 'kein'+kein_style.groups()[1]
    der_form = re.match(r'(der|die|das|dem|den|des)$',word)
    if der_form:
        return word
    der_style = re.match(r'(diese|jede|jene)([rsnm]?$)',word)
    if der_style:
        return 'diese'+der_style.groups()[1]
    return None

class NounsNBC():

    def __init__(self, path):

        with open(path,'rb') as f:

            data = pickle.load(f)
            
            self.nbc_clf = data['clf']
            self.features_encoder = data['features_encoder']
            self.features_list = data['features_list']
            self.rules_list = data['rules_list']
            self.n_last = data['n_last']
        
    def __call__(self, word, constraints=None):
        
        word_parts = [word[idx:] for idx in range(-self.n_last,0)]

        if not constraints:
            constraints = ((-1,-1),)

        data = [word_parts+list(constraint) for constraint in constraints]

        word_enc = self.features_encoder.transform(data).astype(int)
        
        if len(constraints)==1:

            pred = self.nbc_clf.predict(word_enc)[0]
            
        else:
                        
            pred = self.nbc_clf.predict_proba(word_enc).mean(0).argmax()
        
        rule = self.rules_list[pred]
    
        if rule=='-':
            return None
        else:
            seq_to_remove,seq_to_add = rule
            return re.sub(f'{seq_to_remove}$',seq_to_add,word)
            
        return None

class CategoricalNaiveBayes():

    def __init__(self, kappa=2, epsilon=1e-20):
        
        self.kappa = kappa
        self.epsilon = epsilon

    def _compute_priors_logprobs(self, y):

        priors_probs = [class_counts/len(y) for class_counts in self.class_counts]

        self.priors_logprobs = np.log(priors_probs)
        
    def _compute_loglikelihood(self, X, y):
        
        feature_counts = {feature_idx:np.zeros((self.n_categories[feature_idx]+2,self.n_classes)) for feature_idx in range(self.n_features)}
        
        for features, class_idx in zip(X, y):
            
            for feature_idx,feature_value in enumerate(features):
                
                feature_counts[feature_idx][feature_value,class_idx] += 1

        loglikelihood = {feature_idx:np.zeros((self.n_categories[feature_idx]+2,self.n_classes)) for feature_idx in range(self.n_features)}

        for feature_idx in range(self.n_features):
            loglikelihood[feature_idx] = np.log((feature_counts[feature_idx]+self.epsilon)
                                                          / (np.repeat(self.class_counts[None,...], self.n_categories[feature_idx]+2, axis=0)
                                                            + self.kappa*self.epsilon))

            loglikelihood[feature_idx][-1,:] = 0

        self.loglikelihood = loglikelihood

        
    def fit(self, X_train, y_train, priors_logprobs=None):

        counter = Counter(y_train)
        
        class_ids, class_counts = zip(*sorted(counter.items()))
        
        self.class_counts = np.array(class_counts)
        self.n_classes = np.max(class_ids)+1

        self.n_features = X_train.shape[1]
        self.n_categories = X_train.max(axis=0)

        if priors_logprobs is None:
            self._compute_priors_logprobs(y_train)
        else:
            self.priors_logprobs = priors_logprobs

        self._compute_loglikelihood(X_train, y_train)

    def _get_bayes_numerator(self, X):

        n_samples = X.shape[0]

        sample_loglikelihood = np.zeros((n_samples,self.n_features,self.n_classes))

        for feature_idx in range(self.n_features):
            
            sample_loglikelihood[:,feature_idx,:] = self.loglikelihood[feature_idx][X[:,feature_idx]] #N_samplesxN_classes

        numerator = sample_loglikelihood.sum(axis=1)  + self.priors_logprobs[None,...]

        return numerator
            
    def predict_proba(self, X):

        numerator = np.exp(self._get_bayes_numerator(X))
        
        probs = numerator/numerator.sum(axis=1,keepdims=True)
                            
        return probs

    def predict(self, X):

        predicted_class_ids = self._get_bayes_numerator(X).argmax(1)

        return predicted_class_ids
        
    def score(self, X, y):

        y_pred = self.predict(X)

        return (y_pred==np.array(y)).mean()

class NounsStatRules():

    def __init__(self, path):

        with open(path,'rb') as f:
            
            data = pickle.load(f)
            
            self.rules_dict = data['rules_dict']
            self.n_last = data['n_last']
            
    def __call__(self,word):

        for idx in range(-self.n_last,0):
            rule =  self.rules_dict[f'last_{abs(idx)}'].get(word[idx:],None)
            if rule:
                seq_to_remove,seq_to_add = rule
                return re.sub(f'{seq_to_remove}$',seq_to_add,word)

        return None

class GLemma():

    """Wiktionary-based German lemmatizer.

    Provides a lemma for a given word given the POS tag:
    NOUN, VERB, ADJ, ADV.

    Parameters
    ----------

    use_nouns_nbc : bool, default=False
        Use Naive Bayes classifier for unknown nouns.
        Slow (not suitable for annotating large corpora), but precise.

    nouns_statrules_acc : int, default=95
        Accuracy for statistical tables, can be (95, 99, 100) 
        When use_nouns_nbc=False, statistical tables are used for unknown nouns to get a lemma based on the ending.

    guess_adj_lemmas : bool, default=True
        Guess adjective lemmas based on most common endings

    wordfreq_csv : str, default=None
        A file with approximate word frequencies. 
        The file must have 2 tab-separated columns for words and their number of occurrences in a corpus.
        When multiple lemmas for a given word form are possible, the most frequent lemma is taken.
        Does not have to be a lemma list.
        
    
    Examples
    --------
    >>> lemmatizer = GLemma('./data', 
                    wordfreq_csv='data/third-party/FrequencyWords/content/2018/de/de_full.txt')
    >>> lemmatizer('vermalt','VERB')
    'vermalen'

    Notes
    -----
    If a verb prefix is separated in the sentence, it should be attached to the root before the lemmatization:
    Ich hole dich ab --> lemmatizer('abhole','VERB')

    By setting nouns_statrules_acc=100, use_nouns_nbc=False,guess_adj_lemmas=False, and wordfreq_csv=None, the lemmatizer only returns
    lemmas that it's 100% sure about.
    """

    def __init__(self, use_nouns_nbc=False, nouns_statrules_acc = 95, guess_adj_lemmas=True, wordfreq_csv=None):

        lemmatizer_data_path = Path(__file__).parent.resolve() / 'data'
        
        with open(lemmatizer_data_path / 'vocab.json', 'rt', encoding='UTF-8') as json_file:
            self.vocab = json.load(json_file)

        if use_nouns_nbc:
            self.nouns_nbc = NounsNBC(lemmatizer_data_path / 'nouns-nbc-top100.pickle')
        else:
            self.nouns_nbc = None

        self.nouns_stat_rules = NounsStatRules(lemmatizer_data_path / f'nouns_stat_rules-{nouns_statrules_acc}.pickle')
    
        if wordfreq_csv:
            self.wordfreq = pd.read_csv(wordfreq_csv, sep=' ', names=['word','freq'])
            self.wordfreq.word = self.wordfreq.word.str.lower()
            self.wordfreq = self.wordfreq.set_index('word').freq.sort_values(ascending=False) #sort by frequency
            self.wordfreq = self.wordfreq.to_dict()
        else:
            self.wordfreq = None

        self.guess_adj_lemmas = guess_adj_lemmas

    def get_most_frequent_word(self, wordlist):
        '''
        Get the most frequent word out of wordlist
        '''
        
        if self.wordfreq:
            freqs = [self.wordfreq.get(word, np.nan) for word in wordlist]
            if all(np.isnan(freqs)):
                return None
            else:
                #if at least one word in the wordrank dictionary
                return wordlist[np.nanargmax(freqs)]
        else:
            return None

    def get_noun_constraints(self, spacy_token):
        '''
        Get noun constraints in the form (genus, declination) based on the preceeding articles or preposition
        e.g. if the noun preceeded by 'durchs' the constraint is ('n','Akkusativ Singular')
        multiple constraints are possible
        '''

        if spacy_token is None:
            return None
        
        ancestors_lemmas = [x.text.lower() for x in spacy_token.ancestors] #hope to find prepositions here, can be fused with articles, e.g. im, durchs
        for ancestors_lemma in ancestors_lemmas:
            ancestors_constraints = prep_constraints.get(ancestors_lemma, None)
            if ancestors_constraints:
                return ancestors_constraints
            
        childeren_lemmas = [x.text.lower() for x in spacy_token.children] #hope to find determiners here, e.g. der, diese, etc.
        for childeren_lemma in childeren_lemmas:
            base_determiner = get_base_determiner(childeren_lemma)#convert determiners to canonical form
            if base_determiner: 
                return article_constraints[base_determiner]
                
        return None

    def filter_verb_lemmas(self, word, lemmas, spacy_token):

        if spacy_token is None:
            return None
            
        if spacy_token.head.lemma_ in ('haben','sein'):
            #Perfekt suspected
            n_hilfsverb = len(set([y for x in lemmas for y in x['via']]))
            if n_hilfsverb>1: #do we really have to choose between sein and haben?
                lemmas = [x for x in lemmas if x['connection']=='Partizip II' 
                          #token head should match the auxiliary verb
                                    and spacy_token.head.lemma_ in x['via']]
        elif spacy_token.head.lemma_=='werden':
            #werden is an auxiliary verb for Passiv or Futur, 
            #the wordform should be Partizip II (Passiv) or the same as lemma (Futur)
            lemmas = [x for x in lemmas if x['connection']=='Partizip II' 
                                or x['lemma']==word]
        else:
            #no evidence for Perfekt or Passiv, so the wordform can't be Partizip II
            lemmas = [x for x in lemmas if not x['connection']=='Partizip II']

        return lemmas
            
    def get_word_lemma(self, word, pos, spacy_token=None):

        lemmas = self.vocab[pos].get(word, None)   

        if not lemmas:
            #maybe old orthography? try to replace ß with ss at the end of the stem 
            newform=re.sub(r'ß($|es$|t?en$|t?e$|t?e?t$|t?est$)',r'ss\1', word) 
            lemmas = self.vocab[pos].get(newform, None)   

        if not lemmas:
            return None
            
        n_unique_lemmas = len(set([x['lemma'] for x in lemmas])) #count unique lemmas, e.g. 'konzentriert' will have 2 records: one for the infinitive and one for the Partizip II
        
        if n_unique_lemmas>1 and spacy_token:

            #multiple lemmas possible for this wordform
            #use Spacy dependency parcer to reduce the possibilities

            if pos=='N':
                #look for noun constraints, e.g. a related article imposes a particular declination, thus a particular wordform
                constraints = self.get_noun_constraints(spacy_token)
                
                if constraints:
                    lemmas = [lemma for lemma in lemmas if (lemma['genus'],lemma['declination']) in constraints]
                
            elif pos=='V':
                lemmas = self.filter_verb_lemmas(word, lemmas, spacy_token)
                
        lemmas = list(set([x['lemma'] for x in lemmas])) #remove all meta info, take unique words
        
        if not lemmas:
            return None
            
        elif len(lemmas)>1:
            #get most frequent lemma, can be very imprecise for frequency dictionaries computed on small datasets 
            return self.get_most_frequent_word(lemmas)
            
        else:
            return lemmas[0]

    def __call__(self, word=None, pos=None, spacy_token=None):

        lemma = None

        if not word:
            word, pos = spacy_token.text, spacy_token.pos_

        word = word.lower()

        for pos_tag in ('N','V','ADJ','ADV'):
            #conver pos to unified pos_tag
            if pos.startswith(pos_tag):
                pos = pos_tag

        if not pos in ('N','V','ADJ','ADV'):
            #lemmatizer works only for nouns, verbs, adjectives, and adverbs
            return None

        if pos=='ADV':
            #first treat adverb as an adjective
            #because Wiktionary dictionary for adverbs is incomplete and almost any adjective in German can be used as an adverb
            lemma = self.get_word_lemma(word, 'ADJ')

        if not lemma:
            lemma = self.get_word_lemma(word, pos, spacy_token=spacy_token)

        if not lemma:
            if pos=='N':
                if self.nouns_nbc:
                    #Naive Bayes classifier: slow, but precise
                    constraints = self.get_noun_constraints(spacy_token)
                    lemma = self.nouns_nbc(word, constraints)
                else:
                    #statistical tables to predict lemma based on endings
                    lemma = self.nouns_stat_rules(word)
            elif pos=='V':
                    #look for known lemma at the end of the word
                    lemma = get_verb_lemma_fwdsearch(word, lambda x:self.get_word_lemma(x, pos, spacy_token=spacy_token), 
                                                     use_longest_subword=True)
            elif pos in 'ADJ' and self.guess_adj_lemmas:
                    #assume most common adjective endings
                    lemma = re.sub('e[rsnm]?$','',word)
                
        if pos=='N' and lemma:
            #noun lemmas starts with a capital
            lemma = lemma.title()
            
        return lemma