from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer
)

import sys
sys.path.append('../')

from models import T5FineTuner

class Generator:
    def __init__(self, qgmodel: T5FineTuner, tokenizer: T5Tokenizer):
        self.qgmodel = qgmodel
        self.tokenizer = tokenizer
        print("Generator was created.")

    def generate(self, passage, answer_text, max_len_input, max_len_output, num_beams, num_return_sequences, device):
        source_encoding = self.tokenizer(
            answer_text,
            passage,
            max_length=max_len_input,
            padding='max_length',
            truncation = 'only_second',
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )

        # Put this in GPU (faster than using cpu)
        input_ids = source_encoding['input_ids'].to(device)
        attention_mask = source_encoding['attention_mask'].to(device)

        generated_ids = self.qgmodel.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_return_sequences=num_return_sequences, # defaults to 1
            num_beams=num_beams, # defaults to 1 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! myabe experiment with 5
            max_length=max_len_output,
            repetition_penalty=1.0, # defaults to 1.0, #last value was 2.5
            length_penalty=1.0, # defaults to 1.0
            early_stopping=True, # defaults to False
            use_cache=True
        )

        preds = {
            self.tokenizer.decode(generated_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for generated_id in generated_ids
        }

        questions = []
        if isinstance(preds, set):
            for elem in preds:
                questions.append(elem)
        else:
            questions.append(''.join(preds))

        return questions