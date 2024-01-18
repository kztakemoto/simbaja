import openai
import time
import pandas as pd
from tqdm import tqdm

from utils import evaluate_by_chatgpt, get_label

import argparse

#### Parameters #############
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='gpt-3.5-turbo-1106', type=str)
args = parser.parse_args()

def chatgpt(prompt):
    try:
        response = openai.ChatCompletion.create(
            model=args.model,
            messages=[
                {"role": "user", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
    
    except openai.error.OpenAIError as e:
        time.sleep(5)
        response = chatgpt(prompt)

    return response


# load question dataset
questions = pd.read_csv("forbidden_question_set.csv")
questions = questions[questions["response_idx"] == 0]
if "gpt-4" in args.model:
    questions = questions[questions["q_id"] < 10]

# add new columns
questions['response'] = pd.Series(dtype='str')
questions['qa_consistency'] = pd.Series(dtype='str')
questions['label'] = pd.Series(dtype='int')

def save_results():
    questions.to_csv(f"results_MJA_{args.model}.csv", index=False)

for index, row in tqdm(questions.iterrows(), total=len(questions), desc="Processing", ncols=100):
    question = row['question']
    jailbreak_prompt = row['prompt']
    prompt = f"{jailbreak_prompt}\n\n{question}"

    # get response
    response = chatgpt(prompt)
    response = response['choices'][0]['message']['content']

    # evaluate whether this response answer the question
    eval_response = evaluate_by_chatgpt(question, response)
    if eval_response is not None:
        eval_response = eval_response['choices'][0]['message']['content']
    else:
        eval_response = "No"
    
    label = get_label(eval_response)

    questions.at[index, 'response'] = response
    questions.at[index, 'qa_consistency'] = eval_response
    questions.at[index, 'label'] = label

    if index % 100 == 0:
        save_results()

save_results()
