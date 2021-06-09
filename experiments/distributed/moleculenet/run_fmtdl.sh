#!/usr/bin/env bash

CLIENT_NUM=$1
WORKER_NUM=$2
SERVER_NUM=$3
GPU_NUM_PER_SERVER=$4
MODEL=$5
DISTRIBUTION=$6
PARTITION_ALPHA=$7
ROUND=$8
EPOCH=$9
BATCH_SIZE=${10}
LR=${11}
TASK_W=${12}
TASK_W_DECAY=${13}
WD=${14}
HIDDEN_DIM=${15}
NODE_DIM=${16}
DR=${17}
READ_DIM=${18}
GRAPH_DIM=${19}
MASK_TYPE=${20}
DATASET=${21}
DATA_DIR=${22}
CI=${23}

PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM

hostname > mpi_host_file

mpirun -np $PROCESS_NUM -hostfile ./mpi_host_file python3 main_fedgmtl.py \
  --gpu_server_num $SERVER_NUM \
  --gpu_num_per_server $GPU_NUM_PER_SERVER \
  --model $MODEL \
  --dataset $DATASET \
  --data_dir $DATA_DIR \
  --hidden_size $HIDDEN_DIM \
  --node_embedding_dim $NODE_DIM \
  --dropout $DR \
  --mask_type $MASK_TYPE \
  --readout_hidden_dim $READ_DIM \
  --graph_embedding_dim $GRAPH_DIM \
  --partition_method $DISTRIBUTION  \
  --partition_alpha $PARTITION_ALPHA \
  --client_num_in_total $CLIENT_NUM \
  --client_num_per_round $WORKER_NUM \
  --comm_round $ROUND \
  --epochs $EPOCH \
  --batch_size $BATCH_SIZE \
  --task_reg $TASK_W \
  --task_reg_decay $TASK_W_DECAY \
  --wd $WD \
  --lr $LR \
  --ci $CI