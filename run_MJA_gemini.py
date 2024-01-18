import vertexai
from vertexai.preview.generative_models import GenerativeModel

import time
import pandas as pd
from tqdm import tqdm

from utils import evaluate_by_chatgpt, get_label, augmentation
from utils import config_def, safety_settings

import argparse

#### Parameters #############
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='gemini-pro', type=str)
args = parser.parse_args()

vertexai.init(project="YourProject", location="YourLocation")
chat_model = GenerativeModel("gemini-pro")

def gemini(prompt, recursion_count=0):
    if recursion_count >= 5:
        return None

    try:
        chat = chat_model.start_chat()
        res = chat.send_message(prompt,
            generation_config=config_def)
    
    except vertexai.generative_models._generative_models.ResponseBlockedError as e:
        res = gemini(prompt, recursion_count + 1)
    
    except Exception as e:
        time.sleep(5)
        res = gemini(prompt, recursion_count + 1)
    
    return res

# load question dataset
questions = pd.read_csv("forbidden_question_set.csv")
questions = questions[questions["response_idx"] == 0]

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
    response = gemini(prompt)
    if response is not None:
        response = response.candidates[0].content.parts[0].text

        # evaluate whether this response answer the question
        eval_response = evaluate_by_chatgpt(question, response)
        if eval_response is not None:
            eval_response = eval_response['choices'][0]['message']['content']
        else:
            eval_response = "No"

        label = get_label(eval_response)

    else:
        eval_response = None
        label = None

    questions.at[index, 'response'] = response
    questions.at[index, 'qa_consistency'] = eval_response
    questions.at[index, 'label'] = label

    if index % 100 == 0:
        save_results()

save_results()

