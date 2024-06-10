"""Finetuning the library models for sequence classification on GLUE."""

import dataclasses
import logging
import os
import sys
import torch
import random
import transformers
import numpy as np

from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, Optional, List, Union
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import HfArgumentParser, TrainingArguments
from transformers import SchedulerType
from torch.utils.data import DataLoader

from data.dataset import FewShotDataset, data_collator_retrieval
from data.processors import (
    processors_mapping, 
    num_labels_mapping, 
    output_modes_mapping, 
    compute_metrics_mapping, 
    bound_mapping
)
from models.roberta import (
    RobertaForPromptFinetuning, 
    RetrievalAugmentedRoberta
)
from models.trainer import Trainer
from util.args import ModelArguments, DynamicDataTrainingArguments, DynamicTrainingArguments, RetrievalArguments

from filelock import FileLock
from datetime import datetime

from util.hooks import compute_flops

sys.path.append("/root/autodl-tmp/wsy/ReFusion/lib/retriever-lib/src/faisslib")
from retriever import FaissRetriever
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

torch.distributed.init_process_group(backend="nccl", init_method="env://")

transformer_type_mapping = {
    'roberta': 'encoder-only',
    't5': 'encoder-decoder'
}

def setup_parallel():
    rank = torch.distributed.get_rank()
    local_rank = os.environ['LOCAL_RANK']
    master_addr = os.environ['MASTER_ADDR']
    master_port = os.environ['MASTER_PORT']
    logger.info(f"Rank = {rank} is initialized in {master_addr}:{master_port}; local_rank = {local_rank}")
    torch.cuda.set_device(rank)
    return local_rank

def set_seed(seed):
    transformers.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(): 
    parser = HfArgumentParser((ModelArguments, DynamicDataTrainingArguments, DynamicTrainingArguments, RetrievalArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, retrieval_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, retrieval_args = parser.parse_args_into_dataclasses()

    if 'prompt' in model_args.few_shot_type:
        data_args.prompt = True

    if training_args.no_train:
        training_args.do_train = False
    if training_args.no_predict:
        training_args.do_predict = False

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    # Load prompt/template/mapping file
    if data_args.prompt:
        if data_args.prompt_path is not None:
            assert data_args.prompt_id is not None
            prompt_list = []
            with open(data_args.prompt_path) as f:
                for line in f:
                    line = line.strip()
                    template, mapping = line.split('\t')
                    prompt_list.append((template, mapping))

            data_args.template, data_args.mapping = prompt_list[data_args.prompt_id] 
            logger.info("Specify load the %d-th prompt: %s | %s" % (data_args.prompt_id, data_args.template, data_args.mapping))
        else:
            if data_args.template_path is not None:
                with open(data_args.template_path) as f:
                    data_args.template_list = []
                    for line in f:
                        line = line.strip()
                        if len(line) > 0:
                            data_args.template_list.append(line)

                # Load top-n templates
                if data_args.top_n_template is not None:
                    data_args.template_list = data_args.template_list[:data_args.top_n_template]
                logger.info("Load top-%d templates from %s" % (len(data_args.template_list), data_args.template_path))

                # ... or load i-th template
                if data_args.template_id is not None:
                    data_args.template = data_args.template_list[data_args.template_id]
                    data_args.template_list = None
                    logger.info("Specify load the %d-th template: %s" % (data_args.template_id, data_args.template))

            if data_args.mapping_path is not None:
                assert data_args.mapping_id is not None # Only can use one label word mapping
                with open(data_args.mapping_path) as f:
                    mapping_list = []
                    for line in f:
                        line = line.strip()
                        mapping_list.append(line)

                data_args.mapping = mapping_list[data_args.mapping_id]
                logger.info("Specify using the %d-th mapping: %s" % (data_args.mapping_id, data_args.mapping))

    # Check save path
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(f"Output directory ({training_args.output_dir}) already exists.")

    setup_parallel()
    logger.warning(
        "Process rank: %s, world size: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.world_size,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    try:
        num_labels = num_labels_mapping[data_args.task_name]
        output_mode = output_modes_mapping[data_args.task_name]
        logger.info("Task name: {}, number of labels: {}, output mode: {}".format(data_args.task_name, num_labels, output_mode))
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    if retrieval_args.enable_retrieval:
        assert retrieval_args.retrieval_mode != None
        logger.info("Enable Retrievals")
        logger.info(f"{retrieval_args}")
        retriever = FaissRetriever(
            retrieval_args.retriever_path, 
            retrieval_args.nprobe, 
            retrieval_args.topk,
            device_id=retrieval_args.retriever_device
        )
        query_encoder = SentenceTransformer(retrieval_args.encoder_path).to(training_args.device)
    else:
        retriever = None
        query_encoder = None

    # Create config
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )

    if 'prompt' in model_args.few_shot_type:
        if config.model_type == 'roberta':
            if not retrieval_args.enable_retrieval:
                model_fn = RobertaForPromptFinetuning
            else:
                model_fn = RetrievalAugmentedRoberta
        else:
            raise NotImplementedError
    elif model_args.few_shot_type == 'finetune':
        model_fn = AutoModelForSequenceClassification
    else:
        raise NotImplementedError
    model_args.transformer_type = transformer_type_mapping[config.model_type]

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        use_fast=False,
        cache_dir=model_args.cache_dir,
    )

    data_args.return_texts = retrieval_args.enable_retrieval
    logger.info(('Dataset includes texts' if data_args.return_texts else 'Dataset does NOT include text'))

    # Get our special datasets.
    train_dataset = FewShotDataset(
        data_args, 
        tokenizer=tokenizer, 
        mode="train", 
        transformer_type=model_args.transformer_type, 
        retriever=retriever, 
        query_encoder=query_encoder
    )
    eval_dataset = (
        FewShotDataset(
            data_args, 
            tokenizer=tokenizer, 
            mode="dev", 
            transformer_type=model_args.transformer_type, 
            retriever=retriever, 
            query_encoder=query_encoder
        )
        if training_args.do_eval
        else None
    )
    test_dataset = (
        FewShotDataset(
            data_args, 
            tokenizer=tokenizer, 
            mode="test", 
            transformer_type=model_args.transformer_type, 
            retriever=retriever, 
            query_encoder=query_encoder
        )
        if training_args.do_predict
        else None
    )

    model = model_fn.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # Pass dataset and argument information to the model
    if data_args.prompt:
        model.label_word_list = torch.tensor(train_dataset.label_word_list).long().cuda()
    if output_modes_mapping[data_args.task_name] == 'regression':
        # lower / upper bounds
        model.lb, model.ub = bound_mapping[data_args.task_name]
    model.model_args = model_args
    model.data_args = data_args
    model.tokenizer = tokenizer

    if retrieval_args.retrieval_mode == "direct" or retrieval_args.retrieval_mode == "nas":
        logger.info("Wrap retrieval-based modules")
        model.add_retrieval(retrieval_args, retriever, query_encoder)
        model.print_trainable_parameters()

    # Build metricx 
    def build_compute_metrics_fn(task_name: str, transformer_type: str) -> Callable[[EvalPrediction], Dict]:
        def compute_metrics_fn_encoder_only(p: EvalPrediction):
            # Note: the eval dataloader is sequential, so the examples are in order.
            # We average the logits over each sample for using demonstrations.
            predictions = p.predictions
            num_logits = predictions.shape[-1]
            logits = predictions.reshape([eval_dataset.num_sample, -1, num_logits])
            logits = logits.mean(axis=0)
            
            if num_logits == 1:
                preds = np.squeeze(logits)
            else:
                preds = np.argmax(logits, axis=1)

            # Just for sanity, assert label ids are the same.
            label_ids = p.label_ids.reshape([eval_dataset.num_sample, -1])
            label_ids_avg = label_ids.mean(axis=0)
            label_ids_avg = label_ids_avg.astype(p.label_ids.dtype)
            assert (label_ids_avg - label_ids[0]).mean() < 1e-2
            label_ids = label_ids[0]
            
            return compute_metrics_mapping[task_name](task_name, preds, label_ids)
        
        def compute_metrics_fn_encoder_decoder(p: EvalPrediction):
            preds = p.predictions
            label_ids = p.label_ids
            return compute_metrics_mapping[task_name](task_name, preds, label_ids)
        
        if transformer_type == 'encoder-only':
            return compute_metrics_fn_encoder_only
        elif transformer_type == 'encoder-decoder':
            return compute_metrics_fn_encoder_decoder
        else:
            raise NotImplementedError(f'Unsupported transformer type {transformer_type}')
    
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator_retrieval,
        compute_metrics=build_compute_metrics_fn(data_args.task_name, model_args.transformer_type)
    )

    # Training
    if training_args.do_train:
        if retrieval_args.retrieval_mode == 'nas':
            trainer.train_search(
                model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None, 
                transformer_type=model_args.transformer_type,
            )
        else:
            trainer.train(
                model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None, 
                transformer_type=model_args.transformer_type,
            )

    # Evaluation
    final_result = {
        'time': str(datetime.today()),
    }

    eval_results = {}
    if training_args.do_eval:
        logger.info("*** Validate ***")

        eval_datasets = [eval_dataset]

        for eval_dataset in eval_datasets:
            trainer.compute_metrics = build_compute_metrics_fn(eval_dataset.args.task_name, transformer_type=model_args.transformer_type)
            output = trainer.evaluate(eval_dataset=eval_dataset, transformer_type=model_args.transformer_type)
            eval_result = output.metrics 

            output_eval_file = os.path.join(
                training_args.output_dir, f"eval_results_{eval_dataset.args.task_name}.txt"
            )
            if trainer.is_world_process_zero():
                with open(output_eval_file, "w") as writer:
                    logger.info("***** Eval results {} *****".format(eval_dataset.args.task_name))
                    for key, value in eval_result.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))
                        final_result[eval_dataset.args.task_name + '_dev_' + key] = value
            eval_results.update(eval_result)

    test_results = {}
    if training_args.do_predict:
        logger.info("*** Test ***")
        test_datasets = [test_dataset]
        if data_args.task_name == "mnli":
            mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
            test_datasets.append(
                FewShotDataset(
                    mnli_mm_data_args, 
                    tokenizer=tokenizer, 
                    mode="test", 
                    transformer_type=model_args.transformer_type,
                    retriever=retriever, 
                    query_encoder=query_encoder,
                )
            )

        for test_dataset in test_datasets:
            trainer.compute_metrics = build_compute_metrics_fn(test_dataset.args.task_name, transformer_type=model_args.transformer_type)
            output = trainer.evaluate(eval_dataset=test_dataset, transformer_type=model_args.transformer_type)
            test_result = output.metrics

            output_test_file = os.path.join(
                training_args.output_dir, f"test_results_{test_dataset.args.task_name}.txt"
            )
            if trainer.is_world_process_zero():
                with open(output_test_file, "w") as writer:
                    logger.info("***** Test results {} *****".format(test_dataset.args.task_name))
                    for key, value in test_result.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))
                        final_result[test_dataset.args.task_name + '_test_' + key] = value

                if training_args.save_logit:
                    predictions = output.predictions
                    num_logits = predictions.shape[-1]
                    logits = predictions.reshape([test_dataset.num_sample, -1, num_logits]).mean(axis=0)
                    np.save(os.path.join(training_args.save_logit_dir, "{}-{}-{}.npy".format(test_dataset.task_name, training_args.model_id, training_args.array_id)), logits)

            test_results.update(test_result)

    if trainer.is_world_process_zero():
        dataloader = DataLoader(train_dataset, batch_size=1, collate_fn=data_collator_retrieval)
        compute_flops(dataloader, model)

    return eval_results

if __name__ == "__main__":
    main()
