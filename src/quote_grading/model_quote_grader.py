# run `export ANTHROPIC_API_KEY=your_api_key_here` first

import anthropic
import os
import re
import json

# Create the grades directory if it doesn't exist
os.makedirs("./data/grades", exist_ok=True)
output_file = "./data/grades/quotation_scores.json"

completion_file = "./data/completions/model_completions.json"
with open(completion_file, "r") as f:
    model_completions = json.load(f)
    
def quote_classifier_prompt(model_completion):
    return f"""I will give you an output from a language model, and it is up to you to determine how well it can handle opening and closing quotation marks.
            This is part of an ML safety project where we are evaluating model performance on quotation handling.

            If the text appears nonsensical, this shouldn't reduce your quotation score, unless it impacts the model's ability to properly open and close quotes.
            Don't worry if the model is confusing people's names or generating text that doesn't make sense, so long as the ability to open and close quotes remains intact.

            You are to consider the language model generated text, and grade it on a scale from 1 to 10.
            Please output the score via <quotation_score>X</quotation_score> tags.

            Respond with quotation_score tags, nothing else.

            ===BEGIN EXAMPLE ONE===
            Example Language model text:
            Beside the water, a curious young fox approached and asked, "What's the secret to wisdom, old owl?" 
            Hoot, puffing his feathers thoughtfully, replied, "True wisdom, young fox, 
            comes from listening more to the world around you than you speak to it."
            Quotation score:
            <quotation_score>10</quotation_score>
            ===END EXAMPLE ONE===

            ===BEGIN EXAMPLE TWO===
            Example Language model text:
            Beside the water, a curious young fox approached and "asked, What's the secret to wisdom, old owl? 
            Hoot, puffing his feathers thoughtfully, replied, "True wisdom, young fox, 
            comes from listening more to the world" around you than you speak to it".
            Quotation score:
            <quotation_score>2</quotation_score>
            ===END EXAMPLE TWO===

            ===BEGIN EXAMPLE THREE===
            Example Language model text:
            Beside the water, a curious young fox approached and asked, what's the secret to wisdom, old owl? 
            Hoot, puffing his feathers thoughtfully, replied, True wisdom, young fox, 
            comes from listening more to the world" around you than you speak to it.
            Quotation score:
            <quotation_score>2.5</quotation_score>
            ===END EXAMPLE THREE===

            Now that you've seen some examples, here is the text I'd like you to consider.
            As a reminder, you are to consider the language model generated text, and grade it on a scale from 1 to 10.

            Please output the score via <quotation_score>X</quotation_score> tags, where X is a number between 1 and 10.
            Respond with quotation_score tags, nothing else.

            ===BEGIN LANGUAGE MODEL MESSAGE===
            Language model text:
            {model_completion}
            Quotation score:""".strip()

def extract_score(response_text):
    pattern = r'<quotation_score>(.*?)</quotation_score>'
    match = re.search(pattern, response_text)
    if match:
        return float(match.group(1))
    return None

def score_completion(model_completion):
    client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY")
    )
    message = client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": quote_classifier_prompt(model_completion)}
        ]
    )
    content_text = message.content[0].text if isinstance(message.content, list) else message.content
    return extract_score(content_text)


for model_object in model_completions:
    # Extract the completion text from the model object
    # Assuming the JSON contains objects with a "completion" key
    model_name = model_object["model"]
    completions = model_object["completions"]

    for completion_object in completions:
        prompt_text = completion_object["prompt"]
        completion_text = completion_object["completion"]
    
        # Get score for this completion
        score = score_completion(completion_text)

        # Load existing data if the file exists
        data = []
        if os.path.exists(output_file):
            try:
                with open(output_file, "r") as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                # If the file exists but isn't valid JSON, start with empty data
                data = []

        # Add new score entry
        data.append({
            "model": model_name,
            "prompt": prompt_text,
            "completion": completion_text,
            "score": score
        })

        # Write the updated data back to the file
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)