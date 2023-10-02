import os
import lmdb
import json
import faiss
import pickle
import argparse
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

class RetrieverBuilder(object):
    def __init__(
        self, 
        model_path, 
        data_dir=None, 
        output_dir='metadata', 
        batch_size=32, 
        num_sentences_per_file=1000, 
        device_id=-1, 
        verbose=False
    ):
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.verbose = verbose
        self.split_dir = os.path.join(self.output_dir, 'splits')
        self.emb_dir = os.path.join(self.output_dir, 'emb')
        self.db_dir = os.path.join(self.output_dir, 'db')
        self.index_dir = os.path.join(self.output_dir, 'index')
        self.meta_path = os.path.join(self.output_dir, 'metadata.json')
        self.device_id = device_id
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if data_dir != None:
            self._split_sentences(data_dir, num_sentences_per_file)
            self._encode(model_path)
            with open(self.meta_path, 'w') as fout:
                json.dump(self.metadata, fout)
        else:
            assert os.path.exists(self.meta_path)
            with open(self.meta_path, 'r') as fin:
                self.metadata = json.load(fin)
        if self.verbose:
            print(f'Embedding Information {self.metadata}')

    def build(
        self, 
        index_type, 
        train_ratio=0.1, 
        least_num_train=1000000, 
        sub_index_size=1000000
    ):
        # self._build_db()
        self._build_index(
            index_type=index_type, 
            train_ratio=train_ratio, 
            least_num_train=least_num_train, 
            sub_index_size=sub_index_size
        )
    
    def __cut_sentence(self, document):
        sentences = []
        sentence = ''
        for i, w in enumerate(document):
            sentence += w
            if w.endswith('.') or w.endswith('!'):
                if i != len(document) - 1 \
                    and (document[i + 1] == ' ' or document[i + 1] == '\n'):
                    sentences.append(sentence.strip())
                    sentence = ''
                elif i == len(document) -1:
                    sentences.append(sentence.strip())
                    sentence = ''
        if sentence != '':
            sentences.append(sentence.strip())
        return list(filter(lambda x: x.strip() != '', sentences))

    def _split_sentences(self, data_dir, num_sentences_per_file=1000):
        if self.verbose:
            print(f'Split the sentence in {data_dir}, the number of sentences in each split is about {num_sentences_per_file}')

        if not os.path.exists(self.split_dir):
            os.mkdir(self.split_dir)
        
        data_files = glob(os.path.join(data_dir, '*.json'), recursive=True)
        data_files.sort()

        for data_path in data_files:
            num_splits = 0
            all_sentences = []
            basename = data_path.split('/')[-1].removesuffix('.json')
            if self.verbose:
                print(f'Split sentences in {basename}')
            with open(data_path, 'r') as fin:
                for line in fin:
                    line = line.strip()
                    try:
                        data = json.loads(line)
                    except:
                        raise Exception(f"Not a json-formatted row {line}")
                    sentences = self.__cut_sentence(data['text'])
                    if len(all_sentences) + len(sentences) > num_sentences_per_file:
                        num_truncation = num_sentences_per_file - len(all_sentences)
                        all_sentences.extend(sentences[:num_truncation])
                        df = pd.DataFrame({'text': all_sentences})
                        split_path = os.path.join(self.split_dir, f'{basename}-{num_splits}.json')
                        df.to_json(split_path, orient='records', lines=True)
                        num_splits += 1
                        all_sentences = sentences[num_truncation:]
                    else:
                        all_sentences.extend(sentences)
            if len(all_sentences) > 0:
                df = pd.DataFrame({'text': all_sentences})
                split_path = os.path.join(self.split_dir, f'{basename}-{num_splits}.json')
                df.to_json(split_path, orient='records', lines=True)
                num_splits += 1

    def _encode(self, model_path):
        if self.verbose:
            print(f'Encode the data in {self.split_dir}')
        
        if not os.path.exists(self.emb_dir):
            os.mkdir(self.emb_dir)
        
        split_files = glob(os.path.join(self.split_dir, '*.json'), recursive=True)
        split_files.sort()

        if self.device_id >= 0:
            model = SentenceTransformer(model_path, device=f'cuda:{self.device_id}')
        else:
            model = SentenceTransformer(model_path, device=f'cpu')

        self.metadata = {'num_emb': 0, 'emb_dim': 0}
        for i, split_file in enumerate(split_files):
            basename = split_file.split('/')[-1].removesuffix('.json')
            if self.verbose:
                print(f'Encode the split file [{split_file}] ({i + 1}/{len(split_files)})')
            df = pd.read_json(split_file, orient='records', lines=True)
            sentences = df['text'].values
            embeddings = []
            num_batches = int(np.ceil(len(sentences) / self.batch_size))
            for batch_id in tqdm(range(num_batches), total=num_batches):
                batch_l = batch_id * self.batch_size
                batch_r = np.min(((batch_id + 1) * self.batch_size, len(sentences)))
                batch_embs = model.encode(sentences[batch_l:batch_r])
                self.metadata['num_emb'] += int(batch_r - batch_l)
                if self.metadata['emb_dim']:
                    assert self.metadata['emb_dim'] == int(batch_embs.shape[1])
                else:
                    self.metadata['emb_dim'] = int(batch_embs.shape[1])
                embeddings.append(batch_embs)
            embeddings = np.concatenate(embeddings, axis=0).tolist()
            df.insert(0, 'embedding', embeddings)
            emb_path = os.path.join(self.emb_dir, f'{basename}.json.gz')
            df.to_json(emb_path, orient='records', lines=True, compression={'method': 'gzip', 'compresslevel': 5})

    def _build_db(self, map_size=200*1024*1024*1024):
        if self.verbose:
            print(f'Build the database')
        
        if not os.path.exists(self.db_dir):
            os.mkdir(self.db_dir)
        
        env = lmdb.open(self.db_dir, map_size=map_size)
        db_size = 0
        with env.begin(write=False) as txn:
            cursor = txn.cursor()
            for _, __ in cursor:
                db_size += 1
        if self.verbose:
            print(f'Database initial size: {db_size}')
        if db_size != 0:
            if self.verbose:
                print(f'Clear database')
            with env.begin(write=True) as txn:
                cursor = txn.cursor()
                for key, __ in cursor:
                    txn.delete(key)
            db_size = 0

        emb_files = glob(os.path.join(self.emb_dir, '*.json.gz'), recursive=True)
        emb_files.sort()

        for emb_file in tqdm(emb_files):
            df = pd.read_json(emb_file, orient='records', lines=True, compression={'method': 'gzip', 'compresslevel': 5})

            txn = env.begin(write=True)
            for i, row in df.iterrows():
                txn.put(
                str(db_size).encode(),
                pickle.dumps({
                    'text': row['text'],
                    'embedding': row['embedding']
                })
                )
                db_size += 1
            txn.commit()
        env.close()

        self.metadata['db_path'] = self.db_dir
        self.metadata['db_size'] = db_size
        if self.verbose:
            print(f'Database size {db_size}')

    def __sample_train_data(self, train_ratio, least_num_train):
        num_train = int(self.metadata['num_emb'] * train_ratio)
        num_train = np.max((num_train, np.min((self.metadata['num_emb'], least_num_train))))
        
        train_idx = np.arange(self.metadata['num_emb'])
        np.random.shuffle(train_idx)
        train_idx = train_idx[:num_train]
        train_idx.sort()
        
        emb_files = glob(os.path.join(self.emb_dir, '*.json.gz'))
        emb_files.sort()

        train_embs = []
        idx_count = 0
        for emb_file in tqdm(emb_files):
            df = pd.read_json(emb_file, orient='records', lines=True, compression={'method': 'gzip', 'compresslevel': 5})

            embeddings = np.array([d for d in df['embedding'].values]).astype('float32')
            start_idx = idx_count
            end_idx = start_idx + embeddings.shape[0]
            idx = train_idx
            idx = idx[idx >= start_idx]
            idx = idx[idx < end_idx]

            sub_embs = np.take(embeddings, idx - idx_count, axis=0)
            train_embs.append(sub_embs)
            idx_count += embeddings.shape[0]
            assert self.metadata['emb_dim'] == embeddings.shape[1]
            del embeddings
        
        train_embs = np.concatenate(train_embs, axis=0).astype('float32')
        assert train_embs.shape[0] == num_train, f'sample {train_embs.shape[0]} training data but {num_train} data are required'
        assert train_embs.shape[1] == self.metadata['emb_dim']
        return train_embs

    def __train_index(self, train_embs, index_type, trained_index_path):
        index = faiss.index_factory(train_embs.shape[1], index_type)
        if self.device_id >= 0:
            gpu_res = faiss.StandardGpuResources()
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            index = faiss.index_cpu_to_gpu(gpu_res, self.device_id, index, co)
        index.train(train_embs)
        if self.device_id >= 0:
            index = faiss.index_gpu_to_cpu(index)
        faiss.write_index(index, trained_index_path)

    def __build_sub_index(self, buffer_embs, trained_index_path, sub_index_path):
        index = faiss.read_index(trained_index_path)
        if self.device_id >= 0:
            gpu_res = faiss.StandardGpuResources()
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            index = faiss.index_cpu_to_gpu(gpu_res, self.device_id, index, co)
        index.add(buffer_embs)
        if self.device_id >= 0:
            index = faiss.index_gpu_to_cpu(index)
        faiss.write_index(index, sub_index_path)

    def _build_index(
        self, 
        index_type, 
        train_ratio, 
        least_num_train, 
        sub_index_size
    ):
        if self.verbose:
            print(f'Build the index [{index_type}]')
        
        if not os.path.exists(self.index_dir):
            os.mkdir(self.index_dir)

        index_path_prefix = index_type.lower().replace(',', '-').replace('_', '-')
        index_path_prefix = os.path.join(self.index_dir, index_path_prefix)

        trained_index_path = index_path_prefix + '.trained'
        if self.verbose:
            print(f'Sample the training data')
        train_embs = self.__sample_train_data(train_ratio, least_num_train)
        if self.verbose:
            print(f'Train the index on {train_embs.shape[0]} data')
        self.__train_index(train_embs, index_type, trained_index_path)

        if self.verbose:
            print(f'Build the sub-indexes')

        emb_files = glob(os.path.join(self.emb_dir, '*.json.gz'))
        emb_files.sort()

        num_sub_indexes = 0
        buffer_embs = []
        buffer_size = 0
        for i, emb_file in tqdm(enumerate(emb_files), total=len(emb_files)):
            df = pd.read_json(emb_file, orient='records', lines=True, compression={'method': 'gzip', 'compresslevel': 5})
            embeddings = np.array([d for d in df['embedding'].values]).squeeze().astype('float32')
            
            buffer_embs.append(embeddings)
            buffer_size += embeddings.shape[0]
            if buffer_size > sub_index_size or i == len(emb_files) - 1:
                buffer_embs = np.concatenate(buffer_embs, axis=0)
                assert buffer_embs.shape[0] == buffer_size
                assert buffer_embs.shape[1] == self.metadata['emb_dim']
                sub_index_path = index_path_prefix + f'-sub{num_sub_indexes}.index'
                self.__build_sub_index(buffer_embs, trained_index_path, sub_index_path)
                del buffer_embs
                num_sub_indexes = num_sub_indexes + 1
                buffer_embs = []
                buffer_size = 0

        if self.verbose:
            print(f'Merge all sub-indexes')

        trained_index = faiss.read_index(trained_index_path)
        for i in range(num_sub_indexes):
            sub_index_path = index_path_prefix + f'-sub{i}.index'
            sub_index = faiss.read_index(sub_index_path)
            trained_index.merge_from(sub_index, trained_index.ntotal)
            del sub_index
        
        index_path = index_path_prefix + f'.index'
        faiss.write_index(trained_index, index_path)
        self.metadata['index_path'] = index_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_sentences_per_file', type=int, default=1000000)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--train_ratio', type=float, default=0.2)
    parser.add_argument('--least_num_train', type=int, default=1000)
    parser.add_argument('--index_type', type=str, required=True)
    parser.add_argument('--sub_index_size', type=int, default=1000000)
    parser.add_argument('--verbose', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    print(f'Parameters {args}')

    builder = RetrieverBuilder(
        args.model_path, 
        data_dir=args.data_dir, 
        output_dir=args.output_dir, 
        batch_size=args.batch_size, 
        num_sentences_per_file=args.num_sentences_per_file, 
        device_id=args.device_id,
        verbose=args.verbose
    )
    builder.build(
        index_type=args.index_type, 
        train_ratio=args.train_ratio,
        least_num_train=args.least_num_train,
        sub_index_size=args.sub_index_size
    )
