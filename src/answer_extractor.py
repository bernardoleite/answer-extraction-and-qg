import sys
sys.path.append('../')

from keybert import KeyBERT
import spacy
import stanza

class KeyBERTAnswerExtractor:
    def __init__(self, agent_name):
        self.agent_name = agent_name
        self.key_model = KeyBERT()
        print("KeyBERT answer extractor was successfully created.")

    def extract_answers(self, passage, ngram_range, stop_words):
        keywords = self.key_model.extract_keywords(passage, keyphrase_ngram_range=(ngram_range[0], ngram_range[1]), stop_words=stop_words)
        if isinstance(keywords, list) and len(keywords) > 0:
            if isinstance(keywords[0], tuple) and len(keywords[0]) == 2:
                top_keywords = keywords[0][0]
                return top_keywords
            else:
                print("Error.")
                return -1
        else:
            print("No Keywords.")
            return -1

    def getAgentType(self):
        return 'ner'

class NerAnswerExtractor:
    def __init__(self, agent_name):
        self.agent_name = agent_name
        self.nlp = stanza.Pipeline(lang='en', processors='tokenize,ner')
        stanza.download('en')

    def extract_answer(self, text):
        doc = self.nlp(text)
        return doc.ents

    def getAgentType(self):
        return 'ner'