import sys
sys.path.append('../')

from collections import Counter
import nltk
from nltk import ngrams

import spacy
import claucy

from keybert import KeyBERT
import stanza
import random
random.seed(42)

class KeyBERTAnswerExtractor:
    def __init__(self, agent_name):
        self.agent_name = agent_name
        self.key_model = KeyBERT()
        print("KeyBERT answer extractor was successfully created.")

    def extract_answers(self, passage, ngram_range, stop_words, max_answers):
        if max_answers > 20:
            max_answers = 20
        keywords = self.key_model.extract_keywords(passage, keyphrase_ngram_range=(ngram_range[0], ngram_range[1]), stop_words=stop_words, use_mmr=True, diversity=0.7, nr_candidates=20, top_n=max_answers)
        if isinstance(keywords, list) and len(keywords) > 0 and len(keywords) <= max_answers:
            return keywords
        elif isinstance(keywords, list) and len(keywords) > max_answers:
            return random.sample(keywords, max_answers)
        else:
            return []

    def getAgentType(self):
        return 'bert'

class NerAnswerExtractor:
    def __init__(self, agent_name):
        self.agent_name = agent_name
        stanza.download('en')
        self.nlp = stanza.Pipeline(lang='en', processors='tokenize,ner', use_gpu=True)

    def extract_answers(self, text, max_answers, remove_duplicates):
        doc = self.nlp(text)
        if remove_duplicates:
            doc.ents = self.remove_duplicates(doc.ents)
        if len(doc.ents) > 0 and len(doc.ents) <= max_answers:
            return doc.ents
        elif len(doc.ents) > max_answers:
            return random.sample(doc.ents, max_answers)
        else:
            return []

    #https://stackoverflow.com/questions/43319409/remove-duplicates-key-from-list-of-dictionaries-python
    def remove_duplicates(self, doc_ents):
        done = set()
        result = []
        for d in doc_ents:
            if d.text not in done:
                done.add(d.text)  # note it down for further iterations
                result.append(d)
        return result

    def getAgentType(self):
        return 'ner'

class NgramExtractor:
    def __init__(self, agent_name):
        self.agent_name = agent_name

    #https://stackoverflow.com/questions/42373747/is-there-a-more-efficient-way-to-find-most-common-n-grams
    #https://stackoverflow.com/questions/17531684/n-grams-in-python-four-five-six-grams
    def extract_answers(self, text, max_answers):
        ngram_counts = Counter(ngrams(text.split(), 3))
        result_top = ngram_counts.most_common(10)
        print(result_top)
        sys.exit()
        return result_top

    def getAgentType(self):
        return 'ngram'


class ClausieExtractor:
    def __init__(self, agent_name):
        self.agent_name = agent_name
        self.nlp = spacy.load('en_core_web_sm')
        claucy.add_to_pipe(self.nlp) 

    def extract_answers(self, text, max_answers):
        sentences = nltk.sent_tokenize(text)
        all_clauses = []
        for sent in sentences:
            doc = self.nlp(sent)
            num_clauses = len(doc._.clauses)

            idx = 0
            while idx <= num_clauses:
                propositions = -1
                try:
                    propositions = doc._.clauses[idx].to_propositions(as_text=True)
                except:
                    pass
                if isinstance(propositions, list):
                    all_clauses.extend(propositions)
                idx = idx + 1

        if len(all_clauses) <= max_answers:
            return all_clauses
        else:
            return random.sample(all_clauses, max_answers)

    def getAgentType(self):
        return 'clausie'


