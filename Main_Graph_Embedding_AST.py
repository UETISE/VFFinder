import argparse
import os
from os import listdir
from os.path import isfile, join
import torch
import pandas
from graph_embedding.relational_graph import *
import sys


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--node_graph_dir', type=str, help='dir of the node files')
    parser.add_argument('--edge_graph_dir', type=str, help='dir of the edge files')
    parser.add_argument('--embedding_graph_dir', type=str, help='dir to save graph')
    parser.add_argument('--label', type=int, help='label of the commits, 1 if the commits are buggy, 0 otherwise')
    args = parser.parse_args()
    node_graph_dir = args.node_graph_dir
    edge_graph_dir = args.edge_graph_dir
    embedding_graph_dir = args.embedding_graph_dir
    if not os.path.isdir(embedding_graph_dir):
        os.makedirs(embedding_graph_dir)
    label = int(args.label)
    if label == 1:
        embedding_graph_dir = os.path.join(embedding_graph_dir, "VTC")
    else:
        embedding_graph_dir = os.path.join(embedding_graph_dir, "VFC")
    if not os.path.isdir(embedding_graph_dir):
        os.makedirs(embedding_graph_dir)
    node_files = [f for f in listdir(node_graph_dir) if isfile(join(node_graph_dir, f))]
    cm = set([f.split(".")[0].split("_")[-1] for f in node_files])
    for commit_id in cm:
    # for commit_id in cm:
        try:
            save_path = os.path.join(embedding_graph_dir, "data_{}.pt".format(commit_id))
            if os.path.exists(save_path):
                print(f"embedded : {commit_id}")
                continue
            node_info = pandas.read_csv(join(node_graph_dir, "node_" + commit_id + ".csv"))
            edge_info = pandas.read_csv(join(edge_graph_dir, "edge_" + commit_id + ".csv"))
            edge_info = edge_info[edge_info["etype"] == "AST"]
            embed_graph(commit_id, label, node_info,  edge_info,save_path)
            
        except Exception as e:
            exception_type, exception_object, exception_traceback = sys.exc_info()
            filename = exception_traceback.tb_frame.f_code.co_filename
            line_number = exception_traceback.tb_lineno
            print("Exception type: ", exception_type)
            print("File name: ", filename)
            print("Line number: ", line_number)
            print("exception:" + commit_id)
            print(e)