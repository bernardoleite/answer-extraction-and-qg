from generator import Generator
from answer_extractor import KeyBERTAnswerExtractor, NerAnswerExtractor

from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer
)

import argparse
import sys
import pandas as pd
import uuid
sys.path.append('../')

from models import T5FineTuner

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

    answers_infos = []
    test_df = pd.read_pickle(args.test_df_path)
    for index, row in test_df.iterrows():
        paragraph_id = uuid.uuid4().hex
        # agent NER
        if answer_agent.getAgentType() == 'ner':
            answers = answer_agent.extract_answers(row['context'])
            for ans in answers:
                answer_info = {'paragraph_text': row['context'], 'paragraph_id': paragraph_id, 'answer_text': ans.text, 'answer_type': 'ner', 'answer_subtype': ans.type}
                answers_infos.append(answer_info)
        # agent KeyBert
        elif answer_agent.getAgentType() == 'bert':
            answers = answer_agent.extract_answers(row['context'], ngram_range=(1, 3), stop_words="english")
            for ans in answers:
                answer_info = {'paragraph_text': row['context'], 'paragraph_id': paragraph_id, 'answer_text': ans, 'answer_type': 'bert', 'answer_subtype': 'bert'}
                answers_infos.append(answer_info)
        else:
            print("Error. No agent selected.")
            sys.exit()
        print("uma","\n")
    
    print(len(answers_infos))
    sys.exit()

    #questions = agent_gen.generate(row['context'], ans, args.max_len_input, args.max_len_output, args.num_beams, args.num_return_sequences)
    #print(questions)

if __name__ == '__main__':
    # Initialize the Parser
    parser = argparse.ArgumentParser(description = 'Make question predictions based on agent answer extractions and question generator.')

    # Add arguments
    parser.add_argument('-mn','--model_name', type=str, metavar='', default="t5-base", required=False, help='T5 base model name.')
    parser.add_argument('-cp','--checkpoint_path', type=str, metavar='', default="../models_checkpoints/best-checkpoint.ckpt", required=False, help='Model checkpoint path.')

    parser.add_argument('-tp','--test_df_path', type=str, metavar='', default="../data/du_2017_split/raw/dataframe/test_df.pkl", required=False, help='Test dataframe path.')

    parser.add_argument('-bs','--batch_size', type=int, default=32, metavar='', required=False, help='Batch size.')
    parser.add_argument('-mli','--max_len_input', type=int, metavar='', default=512, required=False, help='Max len input for encoding.')
    parser.add_argument('-mlo','--max_len_output', type=int, metavar='', default=96, required=False, help='Max len output for encoding.')

    parser.add_argument('-nb','--num_beams', type=int, metavar='', default=5, required=False, help='Number of beams.')
    parser.add_argument('-nrs','--num_return_sequences', type=int, metavar='', default=1, required=False, help='Number of returned sequences.')
    parser.add_argument('-rp','--repetition_penalty', type=float, metavar='', default=1.0, required=False, help='Repetition Penalty.')
    parser.add_argument('-lp','--length_penalty', type=float, metavar='', default=1.0, required=False, help='Length Penalty.')

    parser.add_argument('-gpu','--use_gpu', type=str, default="True", metavar='', required=False, help='Use GPU (True) or not (false).')

    parser.add_argument('-a','--answer_agent', type=str, default="ner", metavar='', required=True, help='Agent for extracting answers.')

    # Parse arguments
    args = parser.parse_args()

    # Start training
    run(args)
