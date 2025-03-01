import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import argparse
import os
import json
import math
from generate import generate
import random
ANSWER_TRIGGER = "The answer is"


def readFiles():
  with open(args.filepath, 'r') as f:
    data = f.read()
    data = eval(data)
    return data
def create_demo_text(n_shot=8, cot_flag=True):
    question, chain, answer = [], [], []
    question.append(
        "There are 15 trees in the grove. "
        "Grove workers will plant trees in the grove today. "
        "After they are done, there will be 21 trees. "
        "How many trees did the grove workers plant today?"
    )
    chain.append(
        "There are 15 trees originally. "
        "Then there were 21 trees after some more were planted. "
        "So there must have been 21 - 15 = 6."
    )
    answer.append("6")

    question.append(
        "If there are 3 cars in the parking lot and 2 more cars arrive, "
        "how many cars are in the parking lot?"
    )
    chain.append("There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.")
    answer.append("5")

    question.append(
        "Leah had 32 chocolates and her sister had 42. If they ate 35, "
        "how many pieces do they have left in total?"
    )
    chain.append(
        "Originally, Leah had 32 chocolates. "
        "Her sister had 42. So in total they had 32 + 42 = 74. "
        "After eating 35, they had 74 - 35 = 39."
    )
    answer.append("39")

    question.append(
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason "
        "has 12 lollipops. How many lollipops did Jason give to Denny?"
    )
    chain.append(
        "Jason started with 20 lollipops. Then he had 12 after giving some "
        "to Denny. So he gave Denny 20 - 12 = 8."
    )
    answer.append("8")

    question.append(
        "Shawn has five toys. For Christmas, he got two toys each from his "
        "mom and dad. How many toys does he have now?"
    )
    chain.append(
        "Shawn started with 5 toys. If he got 2 toys each from his mom and "
        "dad, then that is 4 more toys. 5 + 4 = 9."
    )
    answer.append("9")

    question.append(
        "There were nine computers in the server room. Five more computers "
        "were installed each day, from monday to thursday. "
        "How many computers are now in the server room?"
    )
    chain.append(
        "There were originally 9 computers. For each of 4 days, 5 more "
        "computers were added. So 5 * 4 = 20 computers were added. "
        "9 + 20 is 29."
    )
    answer.append("29")

    question.append(
        "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On "
        "wednesday, he lost 2 more. "
        "How many golf balls did he have at the end of wednesday?"
    )
    chain.append(
        "Michael started with 58 golf balls. After losing 23 on tuesday, "
        "he had 58 - 23 = 35. After losing 2 more, "
        "he had 35 - 2 = 33 golf balls."
    )
    answer.append("33")

    question.append(
        "Olivia has $23. She bought five bagels for $3 each. "
        "How much money does she have left?"
    )
    chain.append(
        "Olivia had 23 dollars. "
        "5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. "
        "So she has 23 - 15 dollars left. 23 - 15 is 8."
    )
    answer.append("8")

    # randomize order of the examples ...
    index_list = list(range(len(question)))
    random.shuffle(index_list)

    # Concatenate demonstration examples ...
    demo_text = ""
    for i in index_list[:n_shot]:
        if cot_flag:
            demo_text += (
                "Q: "
                + question[i]
                + "\nA: "
                + chain[i]
                + " "
                + ANSWER_TRIGGER
                + " "
                + answer[i]
                + ".\n\n"
            )
        else:
            demo_text += (
                "Question: "
                + question[i]
                + "\nAnswer: "
                + ANSWER_TRIGGER
                + " "
                + answer[i]
                + ".\n\n"
            )
    return demo_text


def build_prompt(input_text, n_shot, cot_flag):
    demo = create_demo_text(n_shot, cot_flag)
    input_text_prompt = demo + "Q: " + input_text + "\n" + "A:"
    return input_text_prompt

def getBatchPreds():
    inps = readFiles()

    device = 'cuda'
    model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    gen_length = args.gen_len
    steps = args.steps

    res = []
    for inp in tqdm(inps):
      q = build_prompt(inp['question'], 4, cot_flag = False)
      m = [{"role": "user", "content": q}]
      user_input = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
      input_ids = tokenizer(user_input)['input_ids']
      input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

      prompt = input_ids

      out = generate(model, prompt, steps=steps, gen_length=gen_length, block_length=args.block_size, temperature=0., cfg_scale=0., remasking=args.remasking_method)
      answer = tokenizer.batch_decode(out[:, prompt.shape[1]:], skip_special_tokens=True)[0]
      res.append({"question":q, "pred":answer,"answer":inp['answer']})

    with open(args.output_path,'w') as f:
      f.write(str(res))



def main():
  getBatchPreds()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get predictions on data")
    parser.add_argument("--filepath", type=str, required=True, help="Path to jsonl-style dataset (e.g. gsm8k.txt)")
    parser.add_argument("--output_path", type=str, default="output_predictions.txt", help="Path to save results")
    parser.add_argument("--remasking_method", type=str, default="low_confidence", help="Remasking method")

    parser.add_argument("--gen_len", type=int, default=256,required=False, help="Hyperparam: gen_len")
    parser.add_argument("--steps", type=int, default=128, required=False, help="Hyperparam: steps")
    parser.add_argument("--block_size", type=int, default=8,required=False, help="Hyperparam: block_size")
    
    args = parser.parse_args()
    main()
