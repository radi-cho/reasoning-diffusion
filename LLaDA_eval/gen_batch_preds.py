import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import argparse
import os
import json
import math
from generate import generate


def readFiles():
  with open(args.filepath, 'r') as f:
    data = f.read()
    data = eval(data)
    return data

def getBatchPreds():
    inps = readFiles()

    device = 'cuda'
    model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    gen_length = 128
    steps = 128

    res = []
    for inp in tqdm(inps):
      q = inp['question']
      m = [{"role": "user", "content": q}]
      user_input = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
      input_ids = tokenizer(user_input)['input_ids']
      input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

      prompt = input_ids

      out = generate(model, prompt, steps=steps, gen_length=gen_length, block_length=32, temperature=0., cfg_scale=0., remasking=args.remasking_method)
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
    
    args = parser.parse_args()
    main()