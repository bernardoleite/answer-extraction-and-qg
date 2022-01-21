import sys
sys.path.append('../src/')

from auxgen import get_answers, get_questions

from flask import Flask, render_template, request, flash, redirect, url_for

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('home.html')

# Generate Quizz
@app.route('/gen_quizz', methods=['GET', 'POST'])
def gen_quizz():
    if request.method == 'POST':
        nr_answers = request.form['nr_answers']
        nr_quest = request.form['nr_quest']
        text_gen = request.form['text_gen']

        if nr_answers == '' or nr_quest == '' or text_gen == '':
            message = 'Please fill in the fields.'
            return render_template('gen_quizz.html', error=message)

        if 'ner' not in request.form and 'bert' not in request.form and 'clausie' not in request.form:
            message = 'Please, select at least one answer extraction agent.'
            return render_template('gen_quizz.html', error=message)

        extracted_answers = []

        # Get ner answers
        if 'ner' in request.form:
            ner_answers = get_answers("ner", text_gen, max_answers=int(nr_answers), remove_duplicates=True)
            if len(ner_answers) > 0: extracted_answers.extend(ner_answers)

        # Get ner answers
        if 'bert' in request.form:
            bert_answers = get_answers("bert", text_gen, max_answers=int(nr_answers), remove_duplicates=False)
            if len(bert_answers) > 0: extracted_answers.extend(bert_answers)

        # Get ner answers
        if 'clausie' in request.form:
            clausie_answers = get_answers("clausie", text_gen, max_answers=int(nr_answers), remove_duplicates=False)
            if len(clausie_answers) > 0: extracted_answers.extend(clausie_answers)

        # Get questions
        answers_questions = get_questions(text_gen, extracted_answers)

        for elem in answers_questions:
            print(elem['gen_question'])
            print(elem['answer_text'])
            print("\n")

    #flash('You are now registered and can log in.', 'success')
    #message = 'Por favor, selecione uma ou mais perguntas para geração.'
    #return render_template('gen_quizz.html', success=message)
    return render_template('gen_quizz.html') 


if __name__ == '__main__':
    app.secret_key = "secret123"
    app.debug = True
    app.run()