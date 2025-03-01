import torch
import re
import os
import random
from tqdm import tqdm
import argparse
from argparse import ArgumentParser


ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

N_SHOT = 8
COT_FLAG = True
DEBUG = False
ANSWER_TRIGGER = "answer is"

def getData():
  with open(args.pred_data, 'r') as f:
    d = f.read()
    return eval(d)

def extract_answer_from_output(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def is_correct(model_answer, answer):
    gt_answer = extract_answer_from_output(answer)
    assert gt_answer != INVALID_ANS
    return model_answer == gt_answer

def clean_answer(model_pred):
    model_pred = model_pred.lower()
    preds = model_pred.split(ANSWER_TRIGGER.lower())
    answer_flag = True if len(preds) > 1 else False
    if answer_flag:
        # Pick first answer with flag
        pred = preds[1]
    else:
        # Pick last number without flag
        pred = preds[-1]

    pred = pred.replace(",", "")
    pred = [s for s in re.findall(r"-?\d+\.?\d*", pred)]

    if len(pred) == 0:
        return INVALID_ANS

    if answer_flag:
        # choose the first element in list
        pred = pred[0]
    else:
        # choose the last element in list
        pred = pred[-1]

    # (For arithmetic tasks) if a word ends with period, it will be omitted ...
    if pred[-1] == ".":
        pred = pred[:-1]

    return pred


def seed_everything(seed: int):
    import random
    import os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def main():

    seed_everything(args.seed)

    data = getData()

    answers = []
    ps = []
    for sample in tqdm(data):
        model_completion = sample['pred']
        model_answer = clean_answer(model_completion)
        ps.append(model_answer)
        is_cor = is_correct(model_answer, sample["answer"])
        answers.append(is_cor)

    with open(args.output_dir + "/results.txt", "w") as f:
        for i in range(len(answers)):
            print(str(answers[i]) + "  " + ps[i], file=f)
    print(float(sum(answers))/len(answers))

    with open(args.output_dir + "/scores.txt", "w") as f:
        print(
            f"Num of total question: {len(answers)}, "
            f"Correct num: {sum(answers)}, "
            f"Accuracy: {float(sum(answers))/len(answers)}.",
            file=f,
        )
    
if __name__ == "__main__":
    parser = ArgumentParser(description="Perform evaluations on data")
    parser.add_argument("--pred_data", type=str, required=True, help="Path to jsonl-style data (e.g. output_predictions.txt)")
    parser.add_argument("--output_dir", type=str, help="Directory to save results + scores")
    parser.add_argument("--seed", default = 42, type=str, help="Rand seed")

    args = parser.parse_args()
    main()
