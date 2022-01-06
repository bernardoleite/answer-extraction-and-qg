import sys
sys.path.append('../')

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
        self.nlp = stanza.Pipeline(lang='en', processors='tokenize,ner')

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