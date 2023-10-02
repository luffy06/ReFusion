import os
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, required=True)
parser.add_argument('--output_path', type=str, required=True)
parser.add_argument('--raw_text', action=argparse.BooleanOptionalAction)
args = parser.parse_args()

fin = open(args.data_path, 'r')
fout = open(args.output_path, 'w')

row = -1
while True:
  data = fin.readline()
  if data == '':
    break
  if data.strip() == '':
    continue
  row = row + 1
  if args.raw_text:
    data = {'meta': {'filename': args.data_path.split('/')[-1], 'row': str(row)}, 'text': data}
    data = json.dumps(data)
  fout.write(data + '\n')

fin.close()
fout.close()
