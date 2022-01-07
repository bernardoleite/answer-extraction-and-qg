from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer
)
from generator import Generator
from models import T5FineTuner

import argparse
import json
import torch
import random
random.seed(42)

def run(args):
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

    # Put model in gpu (if possible) or cpu (if not possible) for inference purpose
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    qgmodel = qgmodel.to(device)
    print ("Device for inference: ",device)

    # Put model in freeze() and eval() model. Not sure the purpose of freeze
    qgmodel.freeze()
    qgmodel.eval()

    # Create agent for question generation
    agent_gen = Generator(qgmodel, t5_tokenizer)

    # Load json files
    with open(args.paragraphs_path, encoding='utf-8') as file:
        paragraphs_compiled = json.load(file)
    with open("../output/answers/" + args.paragraphs_answers_path + "paragraphs_answers.json", encoding='utf-8') as file:
        paragraphs_answers = json.load(file)

    #paragraphs_answers = random.sample(paragraphs_answers, 15) # to remove!!!!!!!!!!!!!!!!!

    # Generate questions
    printcounter = 0
    paragraphs_answers_questions = []
    for elem in paragraphs_answers:
        questions = agent_gen.generate(elem['paragraph_text'], elem['answer_text'], args.max_len_input, args.max_len_output, args.num_beams, args.num_return_sequences)
        for quest in questions:
            paragraphs_answers_questions.append(
                {'paragraph_id': elem['paragraph_id'], 
                'paragraph_text': elem['paragraph_text'], 
                'answer_text': elem['answer_text'], 
                'answer_type': elem['answer_type'], 
                'answer_subtype': elem['answer_subtype'],
                'gen_question': quest}
                )
        if (printcounter == 400):
            print(str(printcounter) + " paragraphs have been processed.")
            printcounter = 0
        printcounter += 1

    # Save paragraphs and answers to json file

    #https://stackoverflow.com/questions/273192/how-can-i-safely-create-a-nested-directory
    prediction_json_path = "../output/questions/" + args.paragraphs_answers_path
    from pathlib import Path
    Path(prediction_json_path).mkdir(parents=True, exist_ok=True)

    # https://stackoverflow.com/questions/12309269/how-do-i-write-json-data-to-a-file
    with open(prediction_json_path + '/para_answers_questions.json', 'w', encoding='utf-8') as file:
        json.dump(paragraphs_answers_questions, file)

    print("Paragraphs and extracted answers were saved in ", prediction_json_path)

if __name__ == '__main__':
    # Initialize the Parser
    parser = argparse.ArgumentParser(description = 'Make question predictions based on agent answer extractions and question generator.')

    # Add arguments
    parser.add_argument('-mn','--model_name', type=str, metavar='', default="t5-base", required=False, help='T5 base model name.')
    parser.add_argument('-cp','--checkpoint_path', type=str, metavar='', default="../models_checkpoints/best-checkpoint.ckpt", required=False, help='Model checkpoint path.')

    parser.add_argument('-pp','--paragraphs_path', type=str, metavar='', default="../data/paragraphs.json", required=False, help='Paragraphs path.')
    parser.add_argument('-pap','--paragraphs_answers_path', type=str, metavar='', default="ner/2022-01-06_17-08-08/", required=False, help='Paragraphs and answers path.')

    parser.add_argument('-bs','--batch_size', type=int, default=32, metavar='', required=False, help='Batch size.')
    parser.add_argument('-mli','--max_len_input', type=int, metavar='', default=512, required=False, help='Max len input for encoding.')
    parser.add_argument('-mlo','--max_len_output', type=int, metavar='', default=96, required=False, help='Max len output for encoding.')

    parser.add_argument('-nb','--num_beams', type=int, metavar='', default=5, required=False, help='Number of beams.')
    parser.add_argument('-nrs','--num_return_sequences', type=int, metavar='', default=1, required=False, help='Number of returned sequences.')
    parser.add_argument('-rp','--repetition_penalty', type=float, metavar='', default=1.0, required=False, help='Repetition Penalty.')
    parser.add_argument('-lp','--length_penalty', type=float, metavar='', default=1.0, required=False, help='Length Penalty.')

    parser.add_argument('-gpu','--use_gpu', type=str, default="True", metavar='', required=False, help='Use GPU (True) or not (false).')

    # Parse arguments
    args = parser.parse_args()

    # Start training
    run(args)