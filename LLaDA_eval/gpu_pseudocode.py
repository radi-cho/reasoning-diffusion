
## TO RUN PIPELINE
dir = ""
num_gpus = 8
bench_data_path = dir + "gsm8k.txt"
output_dir = dir + "/output"
remasking_method = "low_confidence"

import subprocess

def runsubprocess(cmd):
  subprocess.run(cmd, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

prl_cmd = f"python prep_parallel.py --data {dir + bench_data_path} --output_head {dir + 'chunk'} --num_gpu {num_gpus}"
runsubprocess(prl_cmd)

for i in range(num_gpus):
  ## ADD CODE TO MAKE RUN IN PARALLEL
  pred_cmd = f"python gen_batch_preds.py --filepath {dir + 'chunk' + "_" + str(i) + ".txt"} --output_path {output_dir + "_" + str(i) + "preds.txt"} --remasking_method {remasking_method}"
  runsubprocess(pred_cmd)


## AFTER ALL PREDS ARE GENERATED
for i in range(num_gpus):
  all_pred_data = []
  with open(output_dir + "_" + str(i) + "preds.txt", "r") as f:
    all_pred_data.extend(eval(f.read()))

  with open(output_dir + "_all_preds.txt", "w") as f:
    f.write(str(all_pred_data))

eval_cmd = f"python evaluate_gsm.py --pred_data {output_dir + "_all_preds.txt"} --output_dir {output_dir}"
runsubprocess(eval_cmd)

