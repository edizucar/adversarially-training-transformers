import random

num_examples=30
lower_bound=0
upper_bound=40

with open("prompt.txt","w") as prompt_file:
    txt = ""
    for _ in range(num_examples):
        a = random.randint(lower_bound,upper_bound)
        b = random.randint(lower_bound, upper_bound)
        txt = txt + f"{str(a)}+{str(b)}={str(a+b)},"
    prompt_file.write(txt)