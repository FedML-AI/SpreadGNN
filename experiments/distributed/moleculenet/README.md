# SpreadGNN:  Serverless Multi-task Federated Learning for Graph Neural Networks

This repository is the official implementation of SpreadGNN:  Serverless Multi-task Federated Learning for Graph Neural Networks

## 1. Introduction


Graph Neural Networks (GNNs) are the first choice methods for graph machine learning problems thanks to their ability to learn state-of-the-art level representations from graph-structured data. However, centralizing a massive amount of real-world graph data for GNN training is prohibitive due to user-side privacy concerns, regulation restrictions, and commercial competition. Federated Learning is the de-facto standard for collaborative training of machine learning models over many distributed edge devices without the need for centralization. Nevertheless, training graph neural networks in a federated setting is vaguely defined and brings statistical and systems challenges. This work proposes \texttt{SpreadGNN}, a novel multi-task federated training framework capable of operating in the presence of partial labels and absence of a central server for the first time in the literature. \texttt{SpreadGNN} extends federated multi-task learning to realistic serverless settings for GNNs, and utilizes a novel optimization algorithm with a convergence guarantee, \textit{Decentralized Periodic Averaging SGD (DPA-SGD)}, to solve decentralized multi-task learning problems. We empirically demonstrate the efficacy of our framework on a variety of  non-I.I.D. distributed graph-level molecular property prediction datasets with partial labels. Our results show that  \texttt{SpreadGNN} outperforms GNN models trained over a central server-dependent federated learning system, even in constrained topologies. 



## 2. Installation
<!-- http://doc.fedml.ai/#/installation -->
After `git clone`-ing this repository, please run the following command to install our dependencies.

```bash
conda create -n fedmolecule python=3.7
conda activate fedmolecule
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
conda install -c anaconda mpi4py grpcio 
conda install scikit-learn numpy h5py setproctitle networkx
pip install -r requirements.txt 
cd FedML; git submodule init; git submodule update; cd ../;
pip install -r FedML/requirements.txt
```


## 3. Data Preparation
For each dataset you want to try run the .sh file located in the dataset folder.

For more datasets, visit http://moleculenet.ai/


## 4. Experiments 


### Distributed/Federated Molecule Property Classification experiments
```
sh run_fedavg_distributed_pytorch.sh 6 1 1 1 graphsage homo 150 1 1 0.0015 256 256 0.3 256 256  sider "./../../../data/sider/" 0

##run on background
nohup sh run_fedavg_distributed_pytorch.sh 6 1 1 1 graphsage homo 150 1 1 0.0015 256 256 0.3 256 256  sider "./../../../data/sider/" 0 > ./fedavg-graphsage.log 2>&1 &
```

### Distributed/Federated Molecule Property Regression experiments
```
sh run_fedavg_distributed_reg.sh 6 1 1 1 graphsage homo 150 1 1 0.0015 256 256 0.3 256 256 freesolv "./../../../data/freesolv/" 0

##run on background
nohup sh run_fedavg_distributed_reg.sh 6 1 1 1 graphsage homo 150 1 1 0.0015 256 256 0.3 256 256 freesolv "./../../../data/freesolv/" 0 > ./fedavg-graphsage.log 2>&1 &
```

#### Arguments for Distributed/Federated Training
This is an ordered list of arguments used in distributed/federated experiments. Note, there are additional parameters for this setting.
```
CLIENT_NUM=$1 -> Number of clients in dist/fed setting
WORKER_NUM=$2 -> Number of workers
SERVER_NUM=$3 -> Number of servers
GPU_NUM_PER_SERVER=$4 -> GPU number per server
MODEL=$5 -> Model name
DISTRIBUTION=$6 -> Dataset distribution. homo for IID splitting. hetero for non-IID splitting.
ROUND=$7 -> Number of Distiributed/Federated Learning Rounds
EPOCH=$8 -> Number of epochs to train clients' local models
BATCH_SIZE=$9 -> Batch size 
LR=${10}  -> learning rate
SAGE_DIM=${11} -> Dimenionality of GraphSAGE embedding
NODE_DIM=${12} -> Dimensionality of node embeddings
SAGE_DR=${13} -> Dropout rate applied between GraphSAGE Layers
READ_DIM=${14} -> Dimensioanlity of readout embedding
GRAPH_DIM=${15} -> Dimensionality of graph embedding
DATASET=${16} -> Dataset name (Please check data folder to see all available datasets)
DATA_DIR=${17} -> Dataset directory
CI=${18}
```

### Distributed/Federated Molecule Property Classification with FedGMTL 
```
sh run_fedavg_distributed_pytorch.sh 6 1 1 1 graphsage homo 150 1 1 0.0015 256 256 0.3 256 256  sider "./../../../data/sider/" 0

##run on background
nohup sh run_fedavg_distributed_pytorch.sh 6 1 1 1 graphsage homo 150 1 1 0.0015 256 256 0.3 256 256  sider "./../../../data/sider/" 0 > ./fedavg-graphsage.log 2>&1 &
```

#FedGMTL Classification experiments

```
sh run_fmtdl.sh 8 8 1 1 graphsage hetero 0.5 70 1 1 0.0015 0.3 1 0 64 64 0.3 64 64 1 sider =./../../../data/sider/ 1 0
```

#FedGMTL Regression experiments

```
sh run_fmtdl_reg.sh 8 8 1 1 graphsage hetero 0.5 70 1 1 0.0015 0.3 1 0 64 64 0.3 64 64 1 qm8 "./../../../data/qm8/" 1 0
```

#### Arguments for FedGMTL	
This is an ordered list of arguments used in distributed/federated experiments. Note, there are additional parameters for this setting.
```
CLIENT_NUM=$1 -> Number of clients in dist/fed setting
WORKER_NUM=$2 -> Number of workers
SERVER_NUM=$3 -> Number of servers
GPU_NUM_PER_SERVER=$4 -> GPU number per server
MODEL=$5 -> Model name
DISTRIBUTION=$6 -> Dataset distribution. homo for IID splitting. hetero for non-IID splitting.
PARTITION_ALPHA=$7 -> Alpha parameter for Dirichlet distribution
ROUND=$8 -> Number of Distributed/Federated Learning Rounds
EPOCH=$9 -> Number of epochs to train clients' local models
BATCH_SIZE=${10} -> Batch size 
LR=${11}  -> Learning rate
TASK_W=${12} -> Task-Relationship regularizer weight
TASK_W_DECAY=${13} -> Decay for Task-Relationship regularizer
WD=${14} -> Weight Decay Coefficient
HIDDEN_DIM=${15} -> Dimensionality of GNN Hidden Layer
NODE_DIM=${16}  -> Dimensionality of Node embeddings
DR=${17} -> Dropout rate applied between GraphSAGE Layers
READ_DIM=${18} -> Dimensionality of readout embedding
GRAPH_DIM=${19}  -> Dimensionality of graph embedding
MASK_TYPE=${20} -> Mask scenario (0,1,2)
DATASET=${21} -> Dataset name
DATA_DIR=${22} -> Directory
CI=${23}
```

#SpreadGNN Classification experiments

```
sh run_dfmtdl.sh 8 8 1 1 graphsage hetero 0.5 70 1 1 0.0015 0.3 1 0 64 64 0.3 64 64 1 sider =./../../../data/sider/ 1 0
```

#SpreadGNN Regression experiments

```
sh run_dfmtdl_reg.sh 8 8 1 1 graphsage hetero 0.5 70 1 1 0.0015 0.3 1 0 64 64 0.3 64 64 1 qm8 "./../../../data/qm8/" 1 0
```

#### Arguments for SpreadGNN
This is an ordered list of arguments used in distributed/federated experiments. Note, there are additional parameters for this setting.
```
CLIENT_NUM=$1 -> Number of clients in dist/fed setting
WORKER_NUM=$2 -> Number of workers
SERVER_NUM=$3 -> Number of servers
GPU_NUM_PER_SERVER=$4 -> GPU number per server
MODEL=$5 -> Model name
DISTRIBUTION=$6 -> Dataset distribution. homo for IID splitting. hetero for non-IID splitting.
PARTITION_ALPHA=$7 -> Alpha parameter for Dirichlet distribution
ROUND=$8 -> Number of Distributed/Federated Learning Rounds
EPOCH=$9 -> Number of epochs to train clients' local models
BATCH_SIZE=${10} -> Batch size 
LR=${11}  -> Learning rate
TASK_W=${12} -> Task-Relationship regularizer weight
TASK_W_DECAY=${13} -> Decay for Task-Relationship regularizer
WD=${14} -> Weight Decay Coefficient
HIDDEN_DIM=${15} -> Dimensionality of GNN Hidden Layer
NODE_DIM=${16}  -> Dimensionality of Node embeddings
DR=${17} -> Dropout rate applied between GraphSAGE Layers
READ_DIM=${18} -> Dimensionality of readout embedding
GRAPH_DIM=${19}  -> Dimensionality of graph embedding
MASK_TYPE=${20} -> Mask scenario (0,1,2)
DATASET=${21} -> Dataset name
DATA_DIR=${22} -> Directory
PERIOD=${23} -> Communication Period for Parameter Exchange
CI=${24}
```




## 6. Code Structure of SpreadGNN 
<!-- Note: The code of FedMolecule only uses `FedML/fedml_core` and `FedML/fedml_api`.
In near future, once FedML is stable, we will release it as a python package. 
At that time, we can install FedML package with pip or conda, without the need to use Git submodule. -->

- `FedML`: a soft repository link generated using `git submodule add https://github.com/FedML-AI/FedML`.


- `data`: provide data downloading scripts and store the downloaded datasets.
Note that in `FedML/data`, there also exists datasets for research, but these datasets are used for evaluating federated optimizers (e.g., FedAvg) and platforms.
FedMolecule supports more advanced datasets and models for Molecule Federated Learning.

- `data_preprocessing`: data loaders

- `model`: advanced molecular ML models.

- `trainer`: please define your own `trainer.py` by inheriting the base class in `FedML/fedml-core/trainer/fedavg_trainer.py`.
Some tasks can share the same trainer.

- `experiments/distributed`: 
1. `experiments` is the entry point for training. It contains experiments in different platforms. We start from `distributed`.
1. Every experiment integrates FOUR building blocks `FedML` (federated optimizers), `data_preprocessing`, `model`, `trainer`.
3. To develop new experiments, please refer the code at `experiments/distributed/text-classification`.



## 5. Update FedML Submodule
```
cd FedML
git checkout master && git pull
cd ..
git add FedML
git commit -m "updating submodule FedML to latest"
git push
```


 
