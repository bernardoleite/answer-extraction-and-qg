import sys, time
import argparse

from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer
)
from generator import Generator
from models import T5FineTuner
from answer_extractor import KeyBERTAnswerExtractor, NerAnswerExtractor, ClausieExtractor

import argparse
import torch

def load_generator_args():
    # Initialize the Parser
    parser = argparse.ArgumentParser(description = 'Make question predictions based on agent answer extractions and question generator.')

    # Add arguments
    parser.add_argument('-mn','--model_name', type=str, metavar='', default="t5-base", required=False, help='T5 base model name.')
    parser.add_argument('-cp','--checkpoint_path', type=str, metavar='', default="../models_checkpoints/best-checkpoint.ckpt", required=False, help='Model checkpoint path.')

    parser.add_argument('-pp','--paragraphs_path', type=str, metavar='', default="../data/paragraphs.json", required=False, help='Paragraphs path.')
    parser.add_argument('-pap','--paragraphs_answers_path', type=str, metavar='', default="clausie/2022-01-09_17-28-03/", required=False, help='Paragraphs and answers path.')

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

    return args

def load_generator_model(args):

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
    print ("Device for inference:", device)

    # Put model in freeze() and eval() model. Not sure the purpose of freeze
    qgmodel.freeze()
    qgmodel.eval()

    return qgmodel, t5_tokenizer, device

def get_answers(answer_agent_type, text_gen, max_answers, remove_duplicates):
    extracted_answers = []
    # agent ner
    if answer_agent_type == "ner":
        answer_agent = NerAnswerExtractor("ner")
        answers = answer_agent.extract_answers(text_gen, max_answers=max_answers, remove_duplicates=remove_duplicates)
        for ans in answers:
            new_elem = {'answer_text': ans.text, 'answer_type': 'ner', 'answer_subtype': ans.type}
            extracted_answers.append(new_elem)
    # agent keybert
    elif answer_agent_type == 'bert':
        answer_agent = KeyBERTAnswerExtractor("bert")
        answers = answer_agent.extract_answers(text_gen, ngram_range=(1, 4), stop_words="english", max_answers=max_answers)
        for ans in answers:
            new_elem = {'answer_text': ans[0], 'answer_type': 'bert', 'answer_subtype': 'mmr'}
            extracted_answers.append(new_elem)
    # agent clausie
    elif answer_agent_type == 'clausie':
        answer_agent = ClausieExtractor("clausie")
        answers = answer_agent.extract_answers(text_gen, max_answers=max_answers)
        for ans in answers:
            new_elem = {'answer_text': ans, 'answer_type': 'clausie', 'answer_subtype': 'clausie'}
            extracted_answers.append(new_elem)
    else:
        print("No agent is specified.")

    print("Number of extracted answers:", len(extracted_answers))
    return extracted_answers

def get_questions(text_gen, extracted_answers):
    # Create agent for question generation
    args = load_generator_args()
    qgmodel, t5_tokenizer, device = load_generator_model(args)
    agent_gen = Generator(qgmodel, t5_tokenizer)

    # Generate questions
    start_time_inference = time.time()
    printcounter = 0
    answers_questions = []
    for elem in extracted_answers:
        questions = agent_gen.generate(text_gen, elem['answer_text'], args.max_len_input, args.max_len_output, args.num_beams, args.num_return_sequences, device)
        for quest in questions:
            answers_questions.append(
                {'answer_text': elem['answer_text'], 
                'answer_type': elem['answer_type'], 
                'answer_subtype': elem['answer_subtype'],
                'gen_question': quest}
                )
        if (printcounter == 3):
            print(str(printcounter) + " answer-paragraph pairs have been processed.")
            printcounter = 0
        printcounter += 1

    end_time_inference = time.time()
    total_time_inference = end_time_inference - start_time_inference
    print("Inference time: ", total_time_inference)

    return answers_questions