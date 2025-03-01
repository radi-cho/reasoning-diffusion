## modify these paths accordingly and run -> ~5-7 mins
bench_data_path = "/content/gsmSUBSAMPLE.txt"
output_dir =  "/content/outputs/"
remasking_method = "low_confidence"

import subprocess
def runsubprocess(cmd):
  subprocess.run(cmd, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

pred_cmd = f"python gen_batch_preds.py --filepath {bench_data_path} --output_path {output_dir}preds.txt --remasking_method {remasking_method}"
runsubprocess(pred_cmd)

eval_cmd = f"python evaluate_gsm.py --pred_data {output_dir}preds.txt --output_dir {output_dir}"
runsubprocess(eval_cmd)
