import argparse
import nltk
import json
import sys

#from readability import Readability

def run(args):

# Load json files

    with open(args.para_answers_questions_path, encoding='utf-8') as file:
        para_answers_questions = json.load(file)
        para_answers_questions = para_answers_questions[0:10583] # to be balanced
        
    # first package
    if args.eval_package == "read_1":
        import readability
        para_answers_questions_processed = []
        for paq in para_answers_questions:
            question_processed = nltk.word_tokenize(paq["gen_question"])
            question_processed = ' '.join(question_processed)
            para_answers_questions_processed.append(question_processed+"\n")
            all_text = ''.join(para_answers_questions_processed)

        all_results = readability.getmeasures(all_text, lang='en')
        print("\n")
        print(all_results['readability grades'],"\n")
        print(all_results['sentence info'],"\n")
        print(all_results['word usage'],"\n")
        print(all_results['sentence beginnings'],"\n")

    # second and third packages
    elif args.eval_package == "read_2_3":
        import textstat
        from readability import Readability 
        para_answers_questions_processed = []
        for paq in para_answers_questions:
            para_answers_questions_processed.append(paq["gen_question"])
            my_text = ' '.join(para_answers_questions_processed)
        
        textstat.set_lang("en")
        print("flesch_reading_ease: ", textstat.flesch_reading_ease(my_text))
        print("flesch_kincaid_grade: ", textstat.flesch_kincaid_grade(my_text))
        print("smog_index: ", textstat.smog_index(my_text))
        print("coleman_liau_index: ", textstat.coleman_liau_index(my_text))
        print("automated_readability_index: ", textstat.automated_readability_index(my_text))
        print("dale_chall_readability_score: ", textstat.dale_chall_readability_score(my_text))
        print("difficult_words: ", textstat.difficult_words(my_text))
        print("linsear_write_formula: ", textstat.linsear_write_formula(my_text))
        print("gunning_fog: ", textstat.gunning_fog(my_text))
        print("text_standard: ", textstat.text_standard(my_text))
        #print("fernandez_huerta: ", textstat.fernandez_huerta(my_text))
        #print("szigriszt_pazos: ", textstat.szigriszt_pazos(my_text))
        #print("gutierrez_polini: ", textstat.gutierrez_polini(my_text))
        #print("crawford: ", textstat.crawford(my_text))
        print("gulpease_index: ", textstat.gulpease_index(my_text))
        print("osman: ", textstat.osman(my_text))

        # third package 
        print("\n\n")
        r = Readability(my_text)
        print("flesch_kincaid_grade: ", r.flesch_kincaid())
        print("flesch_reading_ease: ", r.flesch())
        print("gunning_fog: ", r.gunning_fog())
        print("coleman_liau: ", r.coleman_liau())
        print("dale_chall_readability_score: ", r.dale_chall())
        print("ARI: ", r.ari())
        print("linsear_write_formula: ", r.linsear_write())
        print("smog_index: ", r.smog())
        print("spache: ", r.spache())

if __name__ == '__main__':
    # Initialize the Parser
    parser = argparse.ArgumentParser(description = 'Make question predictions based on agent answer extractions and question generator.')

    # Add arguments
    parser.add_argument('-ep','--eval_package', type=str, metavar='', default="textstat", required=True, help='Package used for text statistics.')

    #parser.add_argument('-paq','--para_answers_questions_path', type=str, metavar='', default="../output/questions/ner/2022-01-06_17-08-08/para_answers_questions.json", required=False, help='Paragraphs and answers path.')
    #parser.add_argument('-paq','--para_answers_questions_path', type=str, metavar='', default="../output/questions/bert/2022-01-07_16-22-33/para_answers_questions.json", required=False, help='Paragraphs and answers path.')
    #parser.add_argument('-paq','--para_answers_questions_path', type=str, metavar='', default="../output/questions/clausie/2022-01-09_17-28-03/para_answers_questions.json", required=False, help='Paragraphs and answers path.')
    parser.add_argument('-paq','--para_answers_questions_path', type=str, metavar='', default="../output/questions/predictions.json", required=False, help='Paragraphs and answers path.')

    # Parse arguments
    args = parser.parse_args()

    # Start training
    run(args)