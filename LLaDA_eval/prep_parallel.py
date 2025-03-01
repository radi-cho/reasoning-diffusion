
def getData():
  with open(args.data, 'r') as f:
    d = f.read()
    return eval(d)

def saveChunkedData():
  data = getData()
  size = int(len(data) / int(args.num_gpu))
  splits = [data[i:i + size] for i in range(0, len(data), size)]
  
  for i, split in enumerate(splits):
    with open(args.output_head + "_" + str(i) + ".txt", "w") as f:
      f.write(str(split))

def main():
  saveChunkedData()
  print("Saved split data to " + str([args.output_head + "_" + str(i) + ".txt" for i in range(int(args.num_gpu))]))

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform evaluations on data")
    parser.add_argument("--data", type=str, required=True, help="Path to jsonl-style data (e.g. gsm8k.txt)")
    parser.add_argument("--output_head", type=str, help="Path head to save files to")
    parser.add_argument("--num_gpu", default = 8, type=str, help="Num GPUs to split for")

    args = parser.parse_args()
    main()