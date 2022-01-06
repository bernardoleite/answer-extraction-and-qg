import sys
sys.path.append('../')

from answer_extractor import KeyBERTAnswerExtractor, NerAnswerExtractor

from utils import currentdate
import argparse
import json

import random
random.seed(42)

def run(args):
    # Create agents for answers extraction
    if args.answer_agent == "ner":
        answer_agent = NerAnswerExtractor("agent_2")
    elif args.answer_agent == "bert":
        answer_agent = KeyBERTAnswerExtractor("agent_1")
    else:
        print("No agent selected.")
        sys.exit()

    # Load paragraphs file
    with open(args.paragraphs_path, encoding='utf-8') as file:
        paragraphs_compiled = json.load(file)

    #paragraphs_compiled = random.sample(paragraphs_compiled, 5) # to remove!!!!!!!!!!!!!!!!!

    # Extract answers from paragraphs
    paragraphs_answers = []
    printcounter = 0
    for idx, para in enumerate(paragraphs_compiled):
        # agent NER
        if answer_agent.getAgentType() == 'ner':
            answers = answer_agent.extract_answers(para['paragraph_text'], max_answers = args.max_answers, remove_duplicates=True)
            for ans in answers:
                new_elem = {'paragraph_id': para['paragraph_id'], 'paragraph_text': para['paragraph_text'], 'answer_text': ans.text, 'answer_type': 'ner', 'answer_subtype': ans.type}
                paragraphs_answers.append(new_elem)
        # agent KeyBert
        elif answer_agent.getAgentType() == 'bert':
            answers = answer_agent.extract_answers(para['paragraph_text'], ngram_range=(1, 4), stop_words="english", max_answers = args.max_answers)
            for ans in answers:
                new_elem = {'paragraph_id': para['paragraph_id'], 'paragraph_text': para['paragraph_text'], 'answer_text': ans[0], 'answer_type': 'bert', 'answer_subtype': 'mmr'}
                paragraphs_answers.append(new_elem)
        else:
            print("Error. No agent selected.")
            sys.exit()
        if (printcounter == 500):
            print(str(printcounter) + " paragraphs have been processed.")
            printcounter = 0
        printcounter += 1

    # Save paragraphs and answers to json file

    #https://stackoverflow.com/questions/273192/how-can-i-safely-create-a-nested-directory
    prediction_json_path = "../output/answers/" + answer_agent.getAgentType() + "/" + currentdate()
    from pathlib import Path
    Path(prediction_json_path).mkdir(parents=True, exist_ok=True)

    # https://stackoverflow.com/questions/12309269/how-do-i-write-json-data-to-a-file
    with open(prediction_json_path + '/paragraphs_answers.json', 'w', encoding='utf-8') as file:
        json.dump(paragraphs_answers, file)

    print("Paragraphs and extracted answers were saved in ", prediction_json_path)

if __name__ == '__main__':
    # Initialize the Parser
    parser = argparse.ArgumentParser(description = 'Make question predictions based on agent answer extractions and question generator.')

    # Add arguments
    parser.add_argument('-tp','--paragraphs_path', type=str, metavar='', default="../data/paragraphs.json", required=False, help='Test dataframe path.')

    parser.add_argument('-gpu','--use_gpu', type=str, default="True", metavar='', required=False, help='Use GPU (True) or not (false).')

    parser.add_argument('-a','--answer_agent', type=str, default="ner", metavar='', required=True, help='Agent for extracting answers.')
    parser.add_argument('-ma','--max_answers', type=int, default=5, metavar='', required=True, help='Max answers for extraction')

    # Parse arguments
    args = parser.parse_args()

    # Start answer extraction
    run(args)
