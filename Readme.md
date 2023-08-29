## Instruction to train word2vec model
```
python Main_Word2Vec.py --vocab_file_path="vocab/word2vec.vocabulary.txt" --vector_length=32 --model_dir="model" --model_name="w2v"
```

You can used our <a href="https://drive.google.com/drive/folders/18JlSJaV7jyHOZi985U3cbY63ccoNSvzs?usp=sharing"> pre-trained model </a>


## Instruction to build graph 
You can build Graph by follow to repo <a href="https://github.com/ttrangnguyen/CTGConstruction">CTGConstruction</a>

Our dataset can be found in <a href="https://drive.google.com/drive/folders/18JlSJaV7jyHOZi985U3cbY63ccoNSvzs?usp=sharing">Link</a>

## Instruction to embed features of nodes and edges of the graphs
```
python Main_Graph_Embedding.py --node_graph_dir="Data/Graph/node" --node_graph_dir="Data/Graph/edge" --label=1 --embedding_graph_dir="Data/embedding" 
```

You can used our <a href="https://www.kaggle.com/datasets/thanhvdz/embedding-ast-changes"> embedded graphs </a>

## Instruction to train and test GNN models

```
python Main_VULJIT_Detection.py --graph_dir='Data/Embedding_CTG'  --train_file='Data/data_split/train_time_id.txt' --test_file='Data/data_split/test_time_id.txt'  --model_dir='Model' --GNN_type="GAT"  --model_name="gat" 
```

Download the commit ids in the training and testing sets from <a href="https://www.kaggle.com/datasets/thanhvdz/data-slient-fix"> here </a>

In order to train GNN models, you need to install the required libraries such as torch and pytorch_geometrics

```
# Install required packages.
import os
import torch
os.environ['TORCH'] = torch.__version__
print(torch.__version__)

!pip install -q torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}.html
!pip install -q torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}.html
!pip install -q git+https://github.com/pyg-team/pytorch_geometric.git
```