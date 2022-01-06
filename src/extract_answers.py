import sys
sys.path.append('../')

from generator import Generator
from answer_extractor import KeyBERTAnswerExtractor, NerAnswerExtractor

from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer
)
from models import T5FineTuner

from utils import currentdate
import argparse
import json

import random
random.seed(42)

def run(args):

    text = "Bernardo and Maria is sad because he has a lot of work to do."
    answers = "Bernardo"

    # Create agents for answers extraction

    if args.answer_agent == "ner":
        answer_agent = NerAnswerExtractor("agent_2")
    elif args.answer_agent == "bert":
        answer_agent = KeyBERTAnswerExtractor("agent_1")
    else:
        print("No agent selected.")
        sys.exit()

    # Load args (needed for model init)
    params_dict = dict(
        batch_size = args.batch_size,
        max_len_input = args.max_len_input,
        max_len_output = args.max_len_output
    )
    params = argparse.Namespace(**params_dict)

    # Load T5 base Tokenizer
    t5_tokenizer = T5Tokenizer.from_pretrained(args.model_name)

    # Load T5 base Model
    t5_model = T5ForConditionalGeneration.from_pretrained(args.model_name)

    # Load T5 fine-tuned model for QG
    checkpoint_path = args.checkpoint_path
    qgmodel = T5FineTuner.load_from_checkpoint(checkpoint_path, hparams=params, t5model=t5_model, t5tokenizer=t5_tokenizer)

    # Create agent for question generation
    agent_gen = Generator(qgmodel, t5_tokenizer)


    with open(args.paragraphs_path, encoding='utf-8') as file:
        paragraphs_compiled = json.load(file)

    paragraphs_compiled = random.sample(paragraphs_compiled, 10) # to remove!

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
        if (printcounter == 5):
            print(str(printcounter) + " paragraphs have been processed")
            printcounter = 0
        printcounter += 1

    # Save to json file

    #https://stackoverflow.com/questions/273192/how-can-i-safely-create-a-nested-directory
    prediction_json_path = "../predictions/answers/" + answer_agent.getAgentType() + "/" + currentdate()
    from pathlib import Path
    Path(prediction_json_path).mkdir(parents=True, exist_ok=True)

    # https://stackoverflow.com/questions/12309269/how-do-i-write-json-data-to-a-file
    with open(prediction_json_path + '/paragraphs_answers.json', 'w', encoding='utf-8') as file:
        json.dump(paragraphs_answers, file)

    print("Paragraphs and extracted answers were saved in ", prediction_json_path)

    #questions = agent_gen.generate(row['context'], ans, args.max_len_input, args.max_len_output, args.num_beams, args.num_return_sequences)
    #print(questions)

if __name__ == '__main__':
    # Initialize the Parser
    parser = argparse.ArgumentParser(description = 'Make question predictions based on agent answer extractions and question generator.')

    # Add arguments
    parser.add_argument('-mn','--model_name', type=str, metavar='', default="t5-base", required=False, help='T5 base model name.')
    parser.add_argument('-cp','--checkpoint_path', type=str, metavar='', default="../models_checkpoints/best-checkpoint.ckpt", required=False, help='Model checkpoint path.')

    parser.add_argument('-tp','--paragraphs_path', type=str, metavar='', default="../data/paragraphs.json", required=False, help='Test dataframe path.')

    parser.add_argument('-bs','--batch_size', type=int, default=32, metavar='', required=False, help='Batch size.')
    parser.add_argument('-mli','--max_len_input', type=int, metavar='', default=512, required=False, help='Max len input for encoding.')
    parser.add_argument('-mlo','--max_len_output', type=int, metavar='', default=96, required=False, help='Max len output for encoding.')

    parser.add_argument('-nb','--num_beams', type=int, metavar='', default=5, required=False, help='Number of beams.')
    parser.add_argument('-nrs','--num_return_sequences', type=int, metavar='', default=1, required=False, help='Number of returned sequences.')
    parser.add_argument('-rp','--repetition_penalty', type=float, metavar='', default=1.0, required=False, help='Repetition Penalty.')
    parser.add_argument('-lp','--length_penalty', type=float, metavar='', default=1.0, required=False, help='Length Penalty.')

    parser.add_argument('-gpu','--use_gpu', type=str, default="True", metavar='', required=False, help='Use GPU (True) or not (false).')

    parser.add_argument('-a','--answer_agent', type=str, default="ner", metavar='', required=True, help='Agent for extracting answers.')
    parser.add_argument('-ma','--max_answers', type=int, default=5, metavar='', required=True, help='Max answers for extraction')

    # Parse arguments
    args = parser.parse_args()

    # Start training
    run(args)
