Instructions for grading model performance on handling quotes and generating coherent text
1. Edit `model_names` in `generate_quote_completions.py` with a list of the model names that you'd like to test
2. Edit `completions/model_prompts.json` with the prompts you'd like to run
3. Run `python generate_quote_completions.py` to generate completions for each prompt, for each model
4. View completions in `completions/model_completions.json`
5. Run `python model_quote_grader.py --grade_first 5` to use Claude to grade the performance of the model on the first 5 completions
6. View model scores in `grades/quotation_scores.json`