# Required environment variables:
# DEVICE: the device id for running models
# PORT: the port
# TYPE: finetune / prompt
# TASK: SST-2 / sst-5 / mr / cr / mpqa / subj / trec / CoLA / MNLI / SNLI / QNLI / RTE / MRPC / QQP / STS-B
# BS: batch size (recommendation: 2 / 4 / 8)
# LR: learning rate (recommendation: 1e-5 / 2e-5 / 5e-5)
# SEED: random seed (13 / 21 / 42 / 87 / 100)
# MODEL: pre-trained model name (roberta-*, bert-*), see Transformers model list

PROJECT_DIR=$(dirname "$(dirname "$(dirname "$(realpath "$0")")")")
DEVICE=3 # Cannot support multiple devices
PORT=2222
TYPE=prompt
TASK_LIST=('SST-2') # 'sst-5' 'mr' 'cr' 'mpqa' 'subj' 'trec' 'CoLA' 'MNLI' 'SNLI' 'QNLI' 'RTE' 'MRPC' 'QQP') # 'STS-B')
BS=2
# The initial learning rate for [`AdamW`] optimizer, defaults to 5e-5.
LR=1e-4
ARCH_LR=2e-4
# The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights in [`AdamW`] optimizer, defaults to 0.
WEIGHT_DECAY=1e-4
ARCH_WEIGHT_DECAY=1e-4
# The beta1 hyperparameter for the [`AdamW`] optimizer, defaults to 0.9.
ADAM_BETA1=0.9
ARCH_ADAM_BETA1=0.9
# The beta2 hyperparameter for the [`AdamW`] optimizer, defaults to 0.999.
ADAM_BETA2=0.999
ARCH_ADAM_BETA2=0.999
# The epsilon hyperparameter for the [`AdamW`] optimizer, defaults to 1e-8.
ADAM_EPSILON=1e-8
ARCH_ADAM_EPSILON=1e-8
# Maximum gradient norm (for gradient clipping), defaults to 1.0.
MAX_GRAD_NORM=1.0
ARCH_MAX_GRAD_NORM=1.0
# The scheduler type to use. See the documentation of [`SchedulerType`] for all possible values, defaults to `"linear"`.
LR_SCHEDULER_TYPE="linear"
ARCH_LR_SCHEDULER_TYPE="linear"
# Number of steps used for a linear warmup from 0 to `learning_rate`.
WARMUP_STEPS=100
ARCH_WARMUP_STEPS=100
SEED_LIST=(13) # 21 42 87 100)
MODEL=/root/autodl-tmp/wsy/models/gemma-2b
IFS='/' read -ra ADDR <<< "$MODEL"
MODEL_NAME=${ADDR[-1]}

# Retrieval variables
RETRIEVER_DEVICE=-1
RETRIEVER_PATH=/root/autodl-tmp/wsy/retriever-lib/metadata/wikitext-103-all-bert-large-uncased-gemma-2b
ENCODER_PATH=/root/autodl-tmp/wsy/models/bert-large-uncased
NPROBE=512
TOPK=64

# Number of training instances per label
K=16

# Training epochs
TRAIN_EPOCHS=200

# Validation steps
EVAL_STEP=200

# Gradient accumulation steps
# For medium-sized GPUs (e.g., 2080ti with 10GB memory), they can only take 
# a maximum batch size of 2 when using large-size models. So we use gradient
# accumulation steps to achieve the same effect of larger batch sizes.
GS=1

for SEED in ${SEED_LIST[*]}
do
    for TASK in ${TASK_LIST[*]}
    do
        # Task specific parameters
        # The default length is 128 and the default number of samples is 16.
        # For some tasks, we use longer length or double demo (when using demonstrations, double the maximum length).
        # For some tasks, we use smaller number of samples to save time (because of the large size of the test sets).
        # All those parameters are set arbitrarily by observing the data distributions.
        TASK_EXTRA=""
        case $TASK in
            SST-2)
                TEMPLATE=Does_the_following_sentence_have_a_positive_or_negative_sentiment_?_*sent_0*_The_answer_is*sep**label**eos*
                MAPPING="{'0':'negative','1':'positive'}"
                ;;
            sst-5)
                TEMPLATE=What_sentiment_does_this_sentence_have?_terrible,_bad,_okay,_good_or_great.*sent_0*_The_answer_is*sep**label**eos*
                MAPPING="{0:'terrible',1:'bad',2:'okay',3:'good',4:'great'}"
                ;;
            mr)
                TEMPLATE=Does_the_following_sentence_have_a_positive_or_negative_sentiment_?_*sent_0*_The_answer_is*sep**label**eos*
                MAPPING="{0:'negative',1:'positive'}"
                ;;
            cr)
                TEMPLATE=Does_the_following_sentence_have_a_positive_or_negative_sentiment_?_*sent_0*_The_answer_is*sep**label**eos*
                MAPPING="{0:'negative',1:'positive'}"
                ;;
            mpqa)
                TEMPLATE=Does_the_following_sentence_have_a_positive_or_negative_sentiment_?_*sent_0*_The_answer_is*sep**label**eos*
                MAPPING="{0:'negative',1:'positive'}"
                ;;
            subj)
                TEMPLATE=Is_this_a_subjective_or_objective_description_?_*sent_0*_The_answer_is*sep**label**eos*
                MAPPING="{0:'subjective',1:'objective'}"
                ;;
            trec)
                TEMPLATE=Which_category_best_describes_the_following_question:_*sent_0*_Choose_from_the_following_list:_description,_entity,_expression,_person,_location,_quantity._The_answer_is*sep**label**eos*
                MAPPING="{0:'description',1:'entity',2:'expression',3:'person',4:'location',5:'quantity'}"
                ;;
            CoLA)
                TEMPLATE=The_following_sentence_is_either_"acceptable",_meaning_it_is_grammatically_correct_and_makes_sense,_or_"unacceptable"._Which_is_it?*sent_0*_The_answer_is*sep**label**eos*
                MAPPING="{'0':'unacceptable','1':'acceptable'}"
                ;;
            MNLI)
                TEMPLATE=*sent-_0*_?_,*+sentl_1**sep**label**eos*
                MAPPING="{'contradiction':'No','entailment':'Yes','neutral':'Maybe'}"
                TASK_EXTRA="--max_seq_len 256"
                ;;
            SNLI)
                TEMPLATE=*sent-_0*_?_,*+sentl_1**sep**label**eos*
                MAPPING="{'contradiction':'No','entailment':'Yes','neutral':'Maybe'}"
                TASK_EXTRA="--max_seq_len 256"
                ;;
            QNLI)
                TEMPLATE=*sent-_0*_?_,*+sentl_1**sep**label**eos*
                MAPPING="{'not_entailment':'No','entailment':'Yes'}"
                ;;
            RTE)
                TEMPLATE=*sent-_0*_?_,*+sentl_1**sep**label**eos*
                MAPPING="{'not_entailment':'No','entailment':'Yes'}"
                TASK_EXTRA="--max_seq_len 256 --first_sent_limit 240"
                ;;
            MRPC)
                TEMPLATE=*sent_0*_?_,*+sentl_1**sep**label**eos*
                MAPPING="{'0':'No','1':'Yes'}"
                ;;
            QQP)
                TEMPLATE=*sent_0*_?_,*+sentl_1**sep**label**eos*
                MAPPING="{'0':'No','1':'Yes'}"
                ;;
            STS-B)
                TEMPLATE=*sent_0*_?_,*+sentl_1**sep**label**eos*
                MAPPING="{'0':'No','1':'Yes'}"
                ;;
        esac
        
        # Use a random number to distinguish different trails (avoid accidental overwriting)
        TRIAL_IDTF=$RANDOM
        DATA_DIR=$PROJECT_DIR/dataset/k-shot/$TASK/$K-$SEED
        LOG_DIR=$PROJECT_DIR/results/main/$MODEL_NAME-refusion/$TASK

        if [[ ! -d $LOG_DIR ]]; then
            mkdir -p $LOG_DIR
        fi

        CUDA_VISIBLE_DEVICES=$DEVICE python -m torch.distributed.launch \
            --nproc_per_node=1 \
            --master_port $PORT $PROJECT_DIR/src/run_glue_decoder.py \
            --task_name $TASK \
            --data_dir $DATA_DIR \
            --overwrite_output_dir \
            --do_train \
            --do_eval \
            --do_predict \
            --evaluation_strategy steps \
            --model_name_or_path $MODEL \
            --few_shot_type $TYPE \
            --num_k $K \
            --max_seq_length 512 \
            --max_target_length 3 \
            --per_device_train_batch_size $BS \
            --per_device_eval_batch_size 16 \
            --gradient_accumulation_steps $GS \
            --learning_rate $LR \
            --weight_decay $WEIGHT_DECAY \
            --adam_beta1 $ADAM_BETA1 \
            --adam_beta2 $ADAM_BETA2 \
            --adam_epsilon $ADAM_EPSILON \
            --max_grad_norm $MAX_GRAD_NORM \
            --lr_scheduler_type $LR_SCHEDULER_TYPE \
            --warmup_steps $WARMUP_STEPS \
            --arch_learning_rate $ARCH_LR \
            --arch_weight_decay $ARCH_WEIGHT_DECAY \
            --arch_adam_beta1 $ARCH_ADAM_BETA1 \
            --arch_adam_beta2 $ARCH_ADAM_BETA2 \
            --arch_adam_epsilon $ARCH_ADAM_EPSILON \
            --arch_max_grad_norm $ARCH_MAX_GRAD_NORM \
            --arch_lr_scheduler_type $ARCH_LR_SCHEDULER_TYPE \
            --arch_warmup_steps $ARCH_WARMUP_STEPS \
            --num_train_epochs $TRAIN_EPOCHS \
            --logging_steps $EVAL_STEP \
            --eval_steps $EVAL_STEP \
            --output_dir checkpoints/$TASK-$TYPE-$K-$SEED-$MODEL-$TRIAL_IDTF \
            --seed $SEED \
            --template $TEMPLATE \
            --mapping $MAPPING \
            --enable_retrieval \
            --retrieval_mode nas \
            --target_modules 'k_proj' 'v_proj' \
            --layers_pattern 'model\.layers\.(\d+)\.*' \
            --encoder_path $ENCODER_PATH \
            --retriever_path $RETRIEVER_PATH \
            --nprobe $NPROBE \
            --topk $TOPK \
            --retriever_device $RETRIEVER_DEVICE \
            --fuse_dim 0 \
            --retrieve_texts \
            --full_training \
            --nas_modules 'identity' 'mask_add_cls' 'ordered_mask_add_cls' \
            $TASK_EXTRA \
            > $LOG_DIR/$TYPE-$K-$SEED-$MODEL_NAME.log 2>&1
        rm -rf $PROJECT_DIR/checkpoints/$TASK-$TYPE-$K-$SEED-$MODEL_NAME-$TRIAL_IDTF
    done
done
