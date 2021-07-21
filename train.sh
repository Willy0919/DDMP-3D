EXP_DIR=output
mkdir -p ${EXP_DIR}/log

now=$(date +"%Y%m%d_%H%M%S")
CUDA_VISIBLE_DEVICES=0,1,2,3,7,6,5,4 python scripts/train.py --config=depth_ddmp\
  2>&1 | tee output/log/$now.log
