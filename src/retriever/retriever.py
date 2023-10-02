import os
import lmdb
import json
import time
import faiss
import pickle
import argparse
import numpy as np
import pandas as pd
import logging
logger = logging.getLogger(__name__)
from glob import glob

class Retriever(object):
    def __init__(self, retriever_dir, nprobe, topk=1, device_id=-1):
        with open(os.path.join(retriever_dir, 'metadata.json'), 'r') as fin:
            self.metadata = json.load(fin)
        db_path = self.metadata['db_path'] if 'db_path' in self.metadata else os.path.join(retriever_dir, 'db')
        if 'index_path' in self.metadata:
            index_path = self.metadata['index_path']
        else:
            index_dir = os.path.join(retriever_dir, 'index')
            index_path = glob(os.path.join(index_dir, '*.index'))[0]
        
        self.index = faiss.read_index(index_path, faiss.IO_FLAG_MMAP)
        if device_id >= 0:
            gpu_res = faiss.StandardGpuResources()
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            self.index = faiss.index_cpu_to_gpu(gpu_res, device_id, self.index, co)
        self.index.nprobe = nprobe
        self.topk = topk
        self.env = lmdb.open(db_path, readonly=True, readahead=True)
        self.retrieval_dim = self.metadata['emb_dim']
        self.cache = None

    def __del__(self):
        self.env.close()

    def search(self, xq, return_texts=False):
        distances, ids = self.index.search(xq, self.topk)
        txn = self.env.begin()
        neighbors_emb = []
        neighbors_text = []
        for query_i in range(xq.shape[0]):
            neighbor_emb = []
            neighbor_text = []
            for j in range(self.topk):
                if ids[query_i][j] != -1:
                    key = txn.get(str(ids[query_i][j]).encode())
                    assert key != None, f'Cannot find key {ids[query_i][j]}'
                    value = pickle.loads(key)
                    text = value['text']
                    emb = np.expand_dims(np.array(value['embedding']).squeeze(), axis=0)
                    neighbor_emb.append(emb)
                    if return_texts:
                        neighbor_text.append(text)
            if len(neighbor_emb) < self.topk:
                for j in range(self.topk - len(neighbor_emb)):
                    neighbor_emb.append(np.zeros((1, self.retrieval_dim)))
            neighbor_emb = np.expand_dims(np.concatenate(neighbor_emb, axis=0), axis=0)
            neighbors_emb.append(neighbor_emb)
            if return_texts:
                neighbors_text.append(neighbor_text)
        neighbors_emb = np.concatenate(neighbors_emb, axis=0)
        if return_texts:
            return neighbors_emb, neighbors_text
        else:
            return neighbors_emb
    
    def save_in_cache(self, neighbors):
        self.cache = neighbors
    
    def fetch_from_cache(self):
        return self.cache

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--retriever_dir', type=str, default=None)
    parser.add_argument('--nprobe', type=int, default=500)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--topk', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--verbose', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    retriever = Retriever(args.retriever_dir, args.nprobe, args.device_id)

    emb_files = glob(os.path.join(args.retriever_dir, 'emb/*.json.gz'))
    emb_files.sort()
    
    for emb_file in emb_files:
        df = pd.read_json(emb_file, orient='records', lines=True)
        embeddings = np.array([emb for emb in df['embedding'].values])

        results = []
        start = time.time_ns()
        num_batches = int(np.ceil(embeddings.shape[0] / args.batch_size))
        for batch_id in range(num_batches):
            l = batch_id * args.batch_size
            r = (batch_id + 1) * args.batch_size
            neighbors = retriever.search(embeddings[l:r], args.topk)
            results.append(neighbors)
        end = time.time_ns()
        latency = (end - start) / (num_batches * args.batch_size)
        print(f'Average latency {latency:.2f} ns')

        recall = 0
        results = np.concatenate(results, axis=0)
        distance = np.ones(embeddings.shape[0])
        for i in range(args.topk):
            distance *= (results[:, i, :] - embeddings).sum(axis=1)
        recall = (distance == 0).sum()
        print(f'Recall@{args.topk} {recall / embeddings.shape[0]}')
