PROJECT_DIR=$(dirname "$(dirname "$(realpath "$0")")")
DATASET_DIR=$PROJECT_DIR/dataset
PY_DIR=$PROJECT_DIR/src/data

if [ ! -d "$DATASET_DIR" ]; then
  mkdir -p "$DATASET_DIR"
fi

if [ ! -f "$DATASET_DIR/datasets.tar" ]; then
  wget -P $DATASET_DIR https://nlp.cs.princeton.edu/projects/lm-bff/datasets.tar 
fi
if [ ! -d "$DATASET_DIR/original" ]; then
  tar xvf $DATASET_DIR/datasets.tar -C $DATASET_DIR
fi

python $PY_DIR/generate_k_shot_glue.py --k 16 --data_dir $DATASET_DIR/original --output_dir $DATASET_DIR