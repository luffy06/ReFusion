import os
import argparse
import pickle
import json
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--emb_path', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--text_path', type=str, required=True)
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--verbose', action=argparse.BooleanOptionalAction)
args = parser.parse_args()

if args.verbose:
  print('Load raw text')
with open(args.text_path, 'r') as fin:
  text_data = list(map(lambda x: x.split('\t'), fin.readlines()[1:]))
  text_data.sort(key=lambda x: int(x[0]))
if args.verbose:
  print(f'Total {len(text_data)} texts')

if args.verbose:
  print('Write formatted text data')
data_path = os.path.join(args.data_dir, args.text_path.split('/')[-1].replace('tsv', 'json'))
with open(data_path, 'w') as fout:
  for i, data in tqdm(enumerate(text_data), total=len(text_data)):
    assert int(data[0]) - 1 == i
    line = {'meta': {'id': data[0], 'title': data[2]}, 'text': data[1].replace('\"', '')}
    line = json.dumps(line)
    fout.write(line + '\n')

emb_files = os.listdir(args.emb_path)
emb_files.sort(key=lambda x: int(x.split('/')[-1].removesuffix('.pkl').split('_')[-1]))

emb_dir = os.path.join(args.output_dir, 'emb')
if not os.path.exists(emb_dir):
  os.makedirs(emb_dir)

if args.verbose:
  print('Format embeddings')
num_emb = 0
emb_dim = 0
for emb_file in tqdm(emb_files):
  emb_file = os.path.join(args.emb_path, emb_file)
  embeddings = pickle.load(open(emb_file, 'rb'))
  embeddings = [(np.expand_dims(e[1], axis=0), text_data[int(e[0].removeprefix('wiki:')) - 1][1].replace('\"', '')) for e in embeddings]
  num_emb += len(embeddings)
  for emb, text in embeddings:
    if emb_dim:
      assert emb.shape[1] == emb_dim
    else:
      emb_dim = emb.shape[1]
  embeddings = {'sentence': embeddings}
  emb_file = os.path.join(emb_dir, emb_file.split('/')[-1].replace('pkl', 'bin'))
  fout = open(emb_file, 'wb')
  pickle.dump(embeddings, fout)
  fout.close()
if args.verbose:
  print(f'Total {num_emb} embeddings, feature dim {emb_dim}')