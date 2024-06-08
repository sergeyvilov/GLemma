# Glemma

New German lemmatizer based on Wiktionary.

**Can reliably detect lemmas for nouns (94%), verbs (99%), adjectives (88%), and adverbs (93%).**

GLemma accepts the wordform and a part-of-speech tag (out of 'N','V','ADJ','ADV') and yields the word lemma. Alternatively, a Spacy token can be provided.

## Features

**Glemma is free for all purposes, including commercial!**

Contrary to other popular lemmatizers, we didn't use data from restricted corpora to create GLemma.

Glemma offers a great range of improvements over another popular Wiktionary-based lemmatozer [IWNLP](https://github.com/Liebeck/IWNLP):

* For verbs, nouns, and adjectives, we add all possible flexion forms to the lookup table, including those that are not on the Wiktionary verb page (e.g. *Konjunktiv I,II* verb forms in *Präterium* and derived forms of nouns with adjective declinations), generated according to German grammar rules.

* When multiple lemmas for a given word are possible, the most frequent will be chosen if a word frequency list is provided (doesn't have to be a lemma list). 

* We account for verb forms in main and subordinate clauses, e.g. *Ich mache auf.* and *Es ist schön, dass du aufmachst.* 

* We account for *zu-Infinitiv* forms, e.g. *aufzumachen* for *aufmachen*.

* For unknown verbs, we use the lemma of the longest known verb at the end of the word, e.g. in the verb *hinaufsetzt*, the ending *setzt* is known, so its lemma will be used to derive the correct lemma for *hinaufsetzt*, which is *hinaufsetzen*.

* The accuracy of verb lemmatizer can be improved by providing a Spacy token. This may help to resolve the cases like *Das Buch hat mit sehr gefallen* (the verb lemma should be *gefallen*) and *Das Buch ist vom Tisch gefallen* (the verb lemma should be *fallen*).

* Lemmas for unknown nouns are determined either based on statistical tables (with 95%, 97% and 100% accuracy level) or using a Na⁄ïve Bayes classifier, the latter is more precise but slower.

* The accuracy of noun lemmatizer can be improved by providing a Spacy token. This may help to reduce the range of possible limits by dependeny-based constraints, e.g. only *Dative* declination can be preceeded by *zu*.

* For adverb lemmas, GLemma first scans the adjective dictionary. This helps to get lemmas for comparative and superlative forms, e.g. *besser* should yield *gut*, for both adjectives and adverbs.

## Installation

Dependencies:
```
python==3.10
pandas==1.5.3
numpy==1.26.0
```

If you want to generate new lookup tables yourself, install
```
pip install wiktionary-de-parser
```

And download the latest [Wiktionary dump](https://dumps.wikimedia.org/dewiktionary/latest/dewiktionary-latest-pages-articles-multistream.xml.bz2).

As German word frequency list, one could use [FrequencyWords](https://github.com/hermitdave/FrequencyWords/blob/master/content/2018/de/de_full.txt)

## Usage

```
from glemma.glemma import GLemma

lemmatizer = GLemma('glemma/data', 
                    wordfreq_csv='glemma/data/third-party/FrequencyWords/content/2018/de/de_full.txt',
                    use_nouns_nbc=False)

#lemmatizer(spacy_token=token)

lemmatizer('aufmachen','V')
```

LICENCE: MIT