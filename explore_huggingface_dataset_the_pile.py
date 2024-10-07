import jsonlines
import itertools
import pandas as pd
from pprint import pprint

# The HuggingFace python library to load datasets
import datasets
from datasets import load_dataset

#pretrained_dataset = load_dataset("EleutherAI/pile", split="train", streaming=True)
'''
Nore > Cousera arning:
Sorry, "The Pile" dataset is currently relocating to a new home and so we can't
show you the same example that is in the video. Here is another dataset, the
"Common Crawl" dataset.
'''

# EleutherAI have amny models
# e.g https://huggingface.co/EleutherAI/polyglot-ko-5.8b
# Uncomment to stream dataset
# pretrained_dataset = load_dataset("c4", "en", split="train", streaming=True)


# This prints out wahts in the taring data - its a rendom assortment of text basically
#Uncomment to view first 5 sections of data
# n = 5
# print("Pretrained dataset:")
# top_n = itertools.islice(pretrained_dataset, n)
# for i in top_n:
#   print(i)

# Below the data from a fine tuned dataset  this is tuned on FAQ data
filename = "lamini_docs.jsonl"
instruction_dataset_df = pd.read_json(filename, lines=True)
instruction_dataset_df

#Various ways of formatting your data

examples = instruction_dataset_df.to_dict()
qa = f"Question :{examples['question'][0]}   Answer: {examples['answer'][0]}"
print(qa)

# Ways to format data for training.

if "question" in examples and "answer" in examples:
  qa = examples["question"][0] + examples["answer"][0]
elif "instruction" in examples and "response" in examples:
  qa = examples["instruction"][0] + examples["response"][0]
elif "input" in examples and "output" in examples:
  qa = examples["input"][0] + examples["output"][0]
else:
  qa = examples["text"][0]

# Sometimes its better to structire data more
# Below using a prompt to specify Question and Answer
# Its recommenede to use the structured approach

prompt_template_qa = """### Question:
{question}

### Answer:
{answer}"""

question = examples["question"][0]
answer = examples["answer"][0]

text_with_prompt_template = prompt_template_qa.format(question=question, answer=answer)

num_examples = len(examples["question"])

finetuning_dataset_text_only = []
finetuning_dataset_question_answer = []

for i in range(num_examples):
  question = examples["question"][i]
  answer = examples["answer"][i]

  text_with_prompt_template_qa = prompt_template_qa.format(question=question, answer=answer)
  finetuning_dataset_text_only.append({"text": text_with_prompt_template_qa})

  text_with_prompt_template_q = prompt_template_q.format(question=question)
  finetuning_dataset_question_answer.append({"question": text_with_prompt_template_q, "answer": answer})


# Its usual to store data in jsonl files
with jsonlines.open(f'lamini_docs_processed.jsonl', 'w') as writer:
    writer.write_all(finetuning_dataset_question_answer)

# Or upload to HuggingFace (You'll need access tokens)
finetuning_dataset_name = "lamini/lamini_docs"
finetuning_dataset = load_dataset(finetuning_dataset_name)
print(finetuning_dataset)