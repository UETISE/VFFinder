import os
import gc
from torch import nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
from tqdm import tqdm
import torch
from jitvul_model.graph_dataset import *
import random
from jitvul_model.GATClassifier import *
from jitvul_model.GCN import *
from jitvul_model.GIN import *
# GraphSAGE
from jitvul_model.GraphSAGE import *
import pandas
tqdm.pandas()

def get_model(data,_params, model_type):
    if model_type == "GCN":
        model = CodeJITGCN(in_channels = data.num_node_features, hidden_channels=_params['hidden_size'], dropout = _params['dropout_rate'], num_of_layers = _params["num_of_layers"], graph_readout_func = _params["graph_readout_func"])
    elif model_type == "GAT":
        print('GAT')
        model = GATClassifier(in_channels = data.num_node_features, hidden_channels=_params['hidden_size'], dropout = _params['dropout_rate'], num_of_layers = _params["num_of_layers"], edge_dim = data.edge_attr.size(-1), graph_readout_func = _params["graph_readout_func"])
    elif model_type == "GIN":
        model = CodeJITGIN(in_channels = data.num_node_features, hidden_channels=_params['hidden_size'], dropout = _params['dropout_rate'], num_of_layers = _params["num_of_layers"], graph_readout_func = _params["graph_readout_func"])
    elif model_type == "GraphSAGE":
        model = CodejitGraphSAGE(in_channels = data.num_node_features, hidden_channels=_params['hidden_size'], dropout = _params['dropout_rate'], num_of_layers = _params["num_of_layers"], graph_readout_func = _params["graph_readout_func"])
    else:
        # print("ERROR:: GNN type " + _params['GNN_type'] + " is not supported.")
        return None
    return model

def train_model(graph_path, train_file_path,test_file_path, _params, model_path, starting_epochs = 0):
    torch.manual_seed(12345)
    tmp_file = open(train_file_path, "r").readlines()
    train_files = [f.replace("\n", "") for f in tmp_file]

    train_dataset = GraphDataset(train_files, graph_path)
    _trainLoader = DataLoader(train_dataset, collate_fn=collate_batch, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_epochs = _params['max_epochs']

    data = {}
    for  graph, _, index in _trainLoader:
        data = graph
        print(data)
        break

    model = get_model(data,_params,_params['GNN_type'])
    model.to(device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=_params['lr'], betas=(0.9, 0.999), eps=1e-08)
    print("learning rate : ", optimizer.param_groups[0]['lr'])
    criterion = nn.CrossEntropyLoss()
    starting_epochs += 1
    valid_auc = 0
    last_train_loss = -1
    last_acc = 0
    for e in range(starting_epochs, max_epochs):
        train_loss, acc = train(e, _trainLoader, model, criterion, optimizer, device)
        if last_train_loss == -1 or last_train_loss > train_loss:
            saved_model_path =  os.path.join(os.path.join(os.getcwd(), model_path), _params['model_name'] + ".pt")
            torch.save(model.state_dict(), saved_model_path)
            last_train_loss = train_loss
        gc.collect()


def train(curr_epochs, _trainLoader, model, criterion, optimizer, device):
    train_loss = 0
    correct = 0
    model.train()
    for graph, commit_id, index in _trainLoader:
        if graph.num_nodes > 1500:
            graph =  graph.subgraph(torch.LongTensor(list(range(0, 1500))))
        if index % 500 == 0:
            print("curr: {}".format(index) + " train loss: {}".format(train_loss / (index + 1)) + " acc:{}".format(correct / (index + 1)))
        if device != 'cpu':
            graph = graph.cuda()
        # if graph.y.item() == 1:
        #     target = torch.tensor([[0,1]],dtype=float)
        # else:
        #     target = torch.tensor([[1,0]],dtype=float)
        # #if graph.y 
        target = graph.y
        target = target.to(device)
        
        if graph.num_nodes == 0 or graph.num_edges == 0:
            continue
        out = model(graph.x, graph.edge_index, graph.edge_type, graph.edge_attr)
        loss = criterion(out, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = out.max(1)
        # print('pred',predicted)
        # print('-'*3)
        correct += predicted.eq(target).sum().item()
        del graph.x, graph.edge_index, graph.edge_type, graph.y, graph, predicted, out
    avg_train_loss = train_loss / len(_trainLoader)
    acc = correct / len(_trainLoader)
    print("correct:", correct)
    print("epochs {}".format(curr_epochs) + " train loss: {}".format(avg_train_loss) + " acc: {}".format(acc))
    gc.collect()
    return avg_train_loss, acc


def test_model(graph_path, test_file_path, _params, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tmp_file = open(test_file_path, "r").readlines()
    test_files = [f.replace("\n", "") for f in tmp_file]
    random.shuffle(test_files)
    test_files = test_files

    test_dataset = GraphDataset(test_files, graph_path)
    _testLoader = DataLoader(test_dataset, collate_fn=collate_batch, shuffle=False)

    data = {}
    for  graph, _, index in _testLoader:
        data = graph
        break
    test_model = get_model(data,_params,_params['GNN_type'])
    test_model.load_state_dict(torch.load(os.path.join(os.path.join(os.getcwd(),model_path), _params['model_name'] + ".pt")))
    test_model.eval()
    evaluate_metrics(_params['model_name'], test_model, _testLoader, device)

def evaluate_metrics(model_name, model, _loader, device):
    print('evaluate >')
    write_to_file_results = []
    model.eval()
    model.to(device)
    with torch.no_grad():
        all_predictions, all_targets, all_probs = [], [], []
        for graph, commit_id, index in _loader:
            if graph.num_nodes > 1500:
                graph =  graph.subgraph(torch.LongTensor(list(range(0, 1500))))
            if device != 'cpu':
                graph = graph.cuda()
            target = graph.y
            if graph.num_nodes == 0 or graph.num_edges == 0:
                print(commit_id)
                continue
            

            out = model(graph.x, graph.edge_index, graph.edge_type, graph.edge_attr)
            target = target.cpu().detach().numpy()
            pred = out.argmax(dim=1).cpu().detach().numpy()
            pro_out = out.tolist()[0]
            prob_1 = out.cpu().detach().numpy()[0][1]
            write_to_file_results.append({"commit_id": commit_id, "predict": pred[0], "target": graph.y.item(),"label0":pro_out[0],"label1":pro_out[1]})
            all_probs.append(prob_1)
            all_predictions.append(pred)
            all_targets.append(target)
            del graph.x, graph.edge_index, graph.edge_type, graph.y, graph
        fpr, tpr, _ = roc_curve(all_targets, all_probs)
        auc_score = round(auc(fpr, tpr) * 100, 2)
        acc = round(accuracy_score(all_targets, all_predictions) * 100, 2)
        print(acc)
        precision = round(precision_score(all_targets, all_predictions) * 100, 2)
        f1 = round(f1_score(all_targets, all_predictions) * 100, 2)
        recall = round(recall_score(all_targets, all_predictions) * 100, 2)
        matrix = confusion_matrix(all_targets, all_predictions)
        target_names = ['clean', 'buggy']
        report = classification_report(all_targets, all_predictions, target_names=target_names)
        result = "auc: {}".format(auc_score) + " acc: {}".format(acc) + " precision: {}".format(precision) + " recall: {}".format(recall) + " f1: {}".format(f1) + " \nreport:\n{}".format(report) + "\nmatrix:\n{}".format(matrix)

        print(result)
    df = pandas.DataFrame.from_dict(write_to_file_results)
    df.to_csv (r'Data/result/'+model_name+'.csv', index = True, header=True)
    model.train()