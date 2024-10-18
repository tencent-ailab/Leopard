#!/bin/bash
EVAL_SETTINGS=$1 # direct none cot
HF_PATH=$2

EVAL_DATAS=('mpdocvqa' 'dude' 'slidevqa' 'mirb' 'mmmu' 'mathvista' 'scienceqa' 'textvqa' 'docvqa' 'visualwebbench')
for EVAL_DATA in "${EVAL_DATAS[@]}"; do
    echo "Evaluating: "$EVAL_DATA

    for i in $(seq 0 $(($GPU_NUM - 1))); do
      CUDA_VISIBLE_DEVICES=$i python llava_multiimg_siglip_anyres.py --shard $i --num_shards $GPU_NUM -c $HF_PATH -d $EVAL_DATA -s $EVAL_SETTINGS &
    done
    wait
    python eval_utils.py --checkpoint $HF_PATH --function "group_acc" -d $EVAL_DATA -s $EVAL_SETTINGS
done

python eval_utils.py -f "merge_all_bench_results" -c $HF_PATH
