# SimBAja
This repository contains code used in [*All in How You Ask for It: Simple Black-Box Method for Jailbreak Attacks*](https://doi.org/10.3390/app14093558).

## Terms of use
MIT licensed. Happy if you cite the following paper when utilizing the codes:

Takemoto K (2024) **All in How You Ask for It: Simple Black-Box Method for Jailbreak Attacks.** Appl. Sci. 14, 3558. doi:10.3390/app14093558.

## Requirements
* Python 3.9
```
pip install -r requirements.txt
```
**NOTE:** The scripts require an OpenAI API key. Please obtain your API key by following [OpenAI's instructions](https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key). To run the scripts for Gemini-Pro (`*_gemini.py`), setup is required.
Please refer to [the Google Cloud Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/gemini).

## Data
Download `questions.csv` and `forbidden_question_set.csv` from the GitHub repository [verazuo/jailbreak_llms](https://github.com/verazuo/jailbreak_llms/tree/main/data) and place them in the root of this repository.

## Usage
### Run Simple Black-Box Jailbreak Attacks
For GPT-3.5,
```
python run_Ours_chatgpt.py
```
For GPT-4,
```
python run_Ours_chatgpt.py --model gpt-4-1106-preview
```
For Gemini-Pro,
```
python run_Ours_gemini.py
```
### Run Other Attack Methods
For PAIR,
* `python run_PAIR_chagpt.py` for GPT-3.5
* `python run_PAIR_chagpt.py --model gpt-4-1106-preview` for GPT-4
* `python run_PAIR_gemini.py`

For manual jailbreak attacks
* `python run_MJA_chagpt.py` for GPT-3.5
* `python run_MJA_chagpt.py --model gpt-4-1106-preview` for GPT-4
* `python run_MJA_gemini.py` for Gemini-Pro

### Compute ASR
```
python compute_ASR.py --file results_*.csv
```
