#!/bin/bash
#sh run_pretrain_megatron_llava.sh dsw /workspace/Pai-Megatron-Patch 7B 4 32 1e-3 1e-4 2048 2048 0 bf16 1 1 sel true false true false 100000 /mnt/llava-datasets/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json /mnt/vicuna-ckpts/vicuna-7b-v1.5-to-mg-tp1-pp1 10000000000 100000000 /mnt/output_patch_test
ifconfig eth1 mtu 4200
export GLOO_SOCKET_IFNAME=bond1


# nccl
export NCCL_DEBUG=WARN
NET_TYPE="low" # A100: high V100: low
if [[ "${NET_TYPE}" = "low" ]]; then
    export NCCL_SOCKET_IFNAME=eth1
    export NCCL_IB_GID_INDEX=3
    export NCCL_IB_HCA=mlx5_2:1,mlx5_2:1
    export NCCL_IB_SL=3
    export NCCL_CHECK_DISABLE=1
    export NCCL_P2P_DISABLE=0
    export NCCL_LL_THRESHOLD=16384
    export NCCL_IB_CUDA_SUPPORT=1
else
    export NCCL_IB_TIMEOUT=24
    export NCCL_IB_GID_INDEX=3
    export NCCL_IB_SL=3
    export NCCL_CHECK_DISABLE=1
    export NCCL_P2P_DISABLE=0
    export NCCL_IB_DISABLE=0
    export NCCL_LL_THRESHOLD=16384
    export NCCL_IB_CUDA_SUPPORT=1
    export NCCL_SOCKET_IFNAME=bond1
    export UCX_NET_DEVICES=bond1
    export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6
    export NCCL_COLLNET_ENABLE=0
    export SHARP_COLL_ENABLE_SAT=0
    export NCCL_NET_GDR_LEVEL=2
    export NCCL_IB_QPS_PER_CONNECTION=4
    export NCCL_IB_TC=160
    export NCCL_PXN_DISABLE=1
fi

export MASTER_ADDR=$CHIEF_IP
export MASTER_PORT=6503
NNODES=$HOST_NUM
NODE_RANK=$INDEX


set -e
MEGATRON_PATCH_PATH=/your_path_to/Pai-Megatron-Patch # your path to Pai-Megatron-Patch
MEGATRON_PATH=${MEGATRON_PATCH_PATH}/Megatron-LM-240603
export PYTHONPATH=${MEGATRON_PATH}:${MEGATRON_PATCH_PATH}:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=$HOST_GPU_NUM
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

MODEL_SIZE="8B"
BATCH_SIZE=1
GLOBAL_BATCH_SIZE=128
LR=1e-5
MIN_LR=0
SEQ_LEN=16384
PAD_LEN=16384
EXTRA_VOCAB_SIZE=0
PR=bf16
TP=8
PP=1
AC=full
DO=false
FL=true
SP=true
TE=false

PRETRAIN_CHECKPOINT_PATH=pretrained_models/llava-pretrain-llama3.1-8B-siglip # path to the pretrain model
DATASET_PATH=/your_path_to/multi_image_data_tars # path to the dataset dir
VISION_TOWER=vision_tower/siglip-so400m-14-364-flash-attn2-navit # path to the vision model

NAME="siglip_anyres" # your experiment name
OUTPUT_BASEPATH=saved_models/llava/$NAME # path to save the model

TRAIN_ITERS=10614 # calculate by total_samples/global_batch_size
LR_WARMUP_ITERS=318 # calculate by TRAIN_ITERS*0.03
LR_DECAY_ITERS=${TRAIN_ITERS}
SAVE_INTERVAL=3000


NUM_LAYERS=32
HIDDEN_SIZE=4096
NUM_ATTN_HEADS=32
INTERMEDIATE_SIZE=14336

gqa_options=" \
		    --group-query-attention \
		    --num-query-groups 8"


if [ $AC = full ]; then
    activation_checkpoint_options=" \
    --recompute-num-layers 1 \
		    --recompute-method uniform \
		    --recompute-granularity full"
elif [ $AC = sel ]; then
    activation_checkpoint_options=" \
        --recompute-activations"
elif [ $AC = none ]; then
    activation_checkpoint_options=" \
                    "
fi

if [ $PR = fp16 ]; then
    pr_options=" \
		    --fp16"
elif [ $PR = bf16 ]; then
    pr_options=" \
        --bf16"
elif [ $PR = fp8 ]; then
    pr_options=" \
        --bf16
        --fp8-hybrid \
        --fp8-amax-compute-algo max \
        --fp8-amax-history-len 1024 \
        --transformer-impl transformer_engine"
fi

if [ $DO = true ]; then
    do_options=" \
		    --use-distributed-optimizer"

elif [ $DO = false ]; then
    do_options=" \
                    "
fi

if [ $FL = true ]; then
    flash_options=" \
		    --use-flash-attn"

elif [ $FL = false ]; then
    flash_options=" \
                    "
fi

if [ $TE = true ]; then
    te_options=" \
		    --transformer-impl transformer_engine"

elif [ $TE = false ]; then
    te_options=" \
            --transformer-impl local"
fi

if [ $SP = true ] && [ $TP -gt 1 ]; then
    sp_options=" \
		    --sequence-parallel"

elif [ $SP = false ]; then
    sp_options=" \
                    "
fi

if [ $PRETRAIN_CHECKPOINT_PATH != none ]; then
    load_options=" \
            --load $PRETRAIN_CHECKPOINT_PATH"
fi

mkdir -p "${OUTPUT_BASEPATH}/tensorboard/"
mkdir -p "${OUTPUT_BASEPATH}/checkpoint/"
mkdir -p "${OUTPUT_BASEPATH}/log/"
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
TENSORBOARD_DIR="${OUTPUT_BASEPATH}/tensorboard/${NAME}_${current_time}"
mkdir -p ${TENSORBOARD_DIR}

SAVED_PRETRAIN_CHECKPOINT_PATH="${OUTPUT_BASEPATH}/checkpoint/${NAME}"


megatron_options="  \
        --save ${SAVED_PRETRAIN_CHECKPOINT_PATH} \
        --image-folder ${DATASET_PATH}/data_json.json \
        --vision-tower ${VISION_TOWER} \
        --image-size 364 \
        --patch-size 14 \
        --version None \
        --mm-projector-type mlp2x_gelu \
        --image-aspect-ratio anyres \
        --train-data-path ${DATASET_PATH} \
        --dataloader-type external \
        --lr ${LR} \
        --min-lr ${MIN_LR} \
        --lr-decay-style cosine \
        --adam-beta1 0.9 \
        --adam-beta2 0.999 \
        --weight-decay 0.0 \
        --clip-grad 1.0 \
        --init-method-std 0.006 \
        --lr-decay-iters ${LR_DECAY_ITERS} \
        --lr-warmup-iters ${LR_WARMUP_ITERS} \
        --train-iters ${TRAIN_ITERS} \
        --micro-batch-size ${BATCH_SIZE} \
        --global-batch-size ${GLOBAL_BATCH_SIZE} \
        --num-layers ${NUM_LAYERS} \
        --hidden-size ${HIDDEN_SIZE} \
        --num-attention-heads ${NUM_ATTN_HEADS} \
        --ffn-hidden-size ${INTERMEDIATE_SIZE} \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings ${SEQ_LEN} \
        --log-interval 1 \
        --eval-interval 10000 \
        --eval-iters 10 \
        --save-interval ${SAVE_INTERVAL} \
        --tensorboard-queue-size 1 \
        --tensorboard-dir ${TENSORBOARD_DIR} \
        --log-timers-to-tensorboard \
        --log-batch-size-to-tensorboard \
        --log-validation-ppl-to-tensorboard \
        --tensor-model-parallel-size ${TP} \
        --pipeline-model-parallel-size ${PP} \
        --num-workers 16 \
        --seed 42 \
        --max-padding-length ${PAD_LEN} \
        --extra-vocab-size ${EXTRA_VOCAB_SIZE} \
        --patch-tokenizer-type LLamaTokenizer \
        --swiglu \
        --normalization RMSNorm \
        --use-rotary-position-embeddings \
        --position-embedding-type rope \
        --untie-embeddings-and-output-weights \
        --disable-bias-linear \
        --rotary-base 500000 \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --no-gradient-accumulation-fusion \
        --dataset LLava-SFT-Raw \
        --answer-loss-only \
        --no-load-optim \
        --no-load-rng \
        --finetune
         "

run_cmd="/root/anaconda3/envs/pai-megatron/bin/python -m torch.distributed.run $DISTRIBUTED_ARGS pretrain_megatron_llava.py
 ${megatron_options} ${pr_options} ${load_options} ${te_options} ${activation_checkpoint_options} ${do_options} ${flash_options} ${sp_options} ${gqa_options}"

LOGS_PATH="${OUTPUT_BASEPATH}/log/${NAME}"
mkdir -p "${LOGS_PATH}"
CURRENT_TIME=$(date +'%m-%d_%T')

echo ${run_cmd}
eval ${run_cmd} 2>&1 | tee -a ${LOGS_PATH}/log_${CURRENT_TIME}.txt
set +x

