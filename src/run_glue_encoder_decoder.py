import random
import logging
import dataclasses
import torch
import transformers
import os
import math
import numpy as np

from tqdm import tqdm
from util.args import ModelArguments, DynamicDataTrainingArguments, DynamicTrainingArguments, RetrievalArguments
from transformers import HfArgumentParser

import sys
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, get_linear_schedule_with_warmup

from transformers import (RobertaConfig, RobertaModel, RobertaTokenizer,
                          BartConfig, BartForConditionalGeneration, BartTokenizer,
                          T5Config, T5ForConditionalGeneration, T5Tokenizer)
MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
                 't5': (T5Config, T5ForConditionalGeneration, T5Tokenizer),
                 'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
                 'bart': (BartConfig, BartForConditionalGeneration, BartTokenizer)}

from data.dataset import FewShotDataset, data_collator_retrieval
from data.processors import (
    processors_mapping, 
    num_labels_mapping, 
    output_modes_mapping, 
    compute_metrics_mapping, 
    bound_mapping
)
from models.t5 import RetrievalAugmentedT5

sys.path.append("/root/autodl-tmp/wsy/ReFusion/lib/retriever-lib/src/faisslib")
from retriever import Retriever
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

transformer_type_mapping = {
    'roberta': 'encoder-only',
    't5': 'encoder-decoder',
    'llama': 'decoder-only'
}


def train(training_args, retrieval_args, train_dataloader, model, optimizer, scheduler):
    tr_loss = 0
    bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
    for step, batch in enumerate(bar):
        batch = {k: v.to(training_args.device) if hasattr(v, "to") else v for k, v in batch.items()}
        source_ids, target_ids = batch["input_ids"], batch["labels"]
        source_mask = batch["attention_mask"]
        target_mask = batch["decoder_attention_mask"]

        if retrieval_args.enable_retrieval:
            outputs = model(
                input_ids=source_ids, 
                attention_mask=source_mask,
                labels=target_ids, 
                decoder_attention_mask=target_mask, 
                neighbors=batch["neighbors"],
                neighbor_texts=batch["neighbor_texts"],
            )
        else:
            outputs = model(
                input_ids=source_ids, 
                attention_mask=source_mask,
                labels=target_ids, 
                decoder_attention_mask=target_mask, 
            )
        loss = outputs.loss
        
        if training_args.n_gpu > 1:
            loss = loss.mean()
        if training_args.gradient_accumulation_steps > 1:
            loss = loss / training_args.gradient_accumulation_steps
        tr_loss += loss.item()
        loss.backward()

        if (step + 1) % training_args.gradient_accumulation_steps == 0:
            # Update parameters
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()


def train_nas(training_args, train_dataloader, eval_dataloader, model, optimizer, scheduler, arch_optimizer, arch_scheduler):
    tr_loss = 0
    bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
    eval_iterator = iter(eval_dataloader)
    for step, batch in enumerate(bar):

        for eval_step in range(training_args.gradient_accumulation_steps):
            try:
                eval_batch = next(eval_iterator)
            except StopIteration:
                eval_iterator = iter(eval_dataloader)
                eval_batch = next(eval_iterator)
            eval_batch = {k: v.to(training_args.device) if hasattr(v, "to") else v for k, v in eval_batch.items()}
            eval_source_ids, eval_target_ids = eval_batch["input_ids"], eval_batch["labels"]
            eval_source_mask = eval_batch["attention_mask"]
            eval_target_mask = eval_batch["decoder_attention_mask"]
            
            eval_outputs = model(
                input_ids=eval_source_ids, 
                attention_mask=eval_source_mask,
                labels=eval_target_ids, 
                decoder_attention_mask=eval_target_mask,
                neighbors=eval_batch["neighbors"],
                neighbor_texts=eval_batch["neighbor_texts"],
            )
            eval_loss = eval_outputs.loss
            if training_args.n_gpu > 1:
                eval_loss = eval_loss.mean()
            if training_args.gradient_accumulation_steps > 1:
                eval_loss = eval_loss / training_args.gradient_accumulation_steps
            eval_loss.backward()
        arch_optimizer.step()
        arch_scheduler.step()
        arch_optimizer.zero_grad()

        batch = {k: v.to(training_args.device) if hasattr(v, "to") else v for k, v in batch.items()}
        source_ids, target_ids = batch["input_ids"], batch["labels"]
        source_mask = batch["attention_mask"]
        target_mask = batch["decoder_attention_mask"]

        outputs = model(
            input_ids=source_ids, 
            attention_mask=source_mask,
            labels=target_ids, 
            decoder_attention_mask=target_mask,
            neighbors=batch["neighbors"],
            neighbor_texts=batch["neighbor_texts"]
        )
        loss = outputs.loss
        
        if training_args.n_gpu > 1:
            loss = loss.mean()
        if training_args.gradient_accumulation_steps > 1:
            loss = loss / training_args.gradient_accumulation_steps
        tr_loss += loss.item()
        loss.backward()

        if (step + 1) % training_args.gradient_accumulation_steps == 0:
            # Update parameters
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()


def evaluation(epoch, training_args, retrieval_args, eval_dataset, model, tokenizer, task, split_tag):
    logger.info("  ***** Running {} evaluation on {} data*****".format(epoch, split_tag))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", training_args.eval_batch_size)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, 
        sampler=eval_sampler, 
        collate_fn=data_collator_retrieval,
        batch_size=training_args.eval_batch_size
    )

    model.eval()
    pred_ids = []
    for batch in eval_dataloader:
        batch = {k: v.to(training_args.device) if hasattr(v, "to") else v for k, v in batch.items()}
        source_ids, target_ids = batch["input_ids"], batch["labels"]
        source_mask = batch["attention_mask"]
        target_mask = batch["decoder_attention_mask"]

        with torch.no_grad():
            if retrieval_args.enable_retrieval:
                if hasattr(model, 'module'):
                    preds = model.module.generate(
                        input_ids=source_ids, 
                        attention_mask=source_mask, 
                        max_length=training_args.max_target_length if training_args.max_target_length else None,
                        neighbors=batch["neighbors"],
                        neighbor_texts=batch["neighbor_texts"],
                    )
                else:
                    preds = model.generate(
                        input_ids=source_ids, 
                        attention_mask=source_mask, 
                        max_length=training_args.max_target_length if training_args.max_target_length else None,
                        neighbors=batch["neighbors"],
                        neighbor_texts=batch["neighbor_texts"],
                    )
            else:
                if hasattr(model, 'module'):
                    preds = model.module.generate(
                        input_ids=source_ids, 
                        attention_mask=source_mask, 
                        max_length=training_args.max_target_length if training_args.max_target_length else None,
                    )
                else:
                    preds = model.generate(
                        input_ids=source_ids, 
                        attention_mask=source_mask, 
                        max_length=training_args.max_target_length if training_args.max_target_length else None,
                    )
            top_preds = list(preds.cpu().numpy())
            pred_ids.extend(top_preds)

    pred_nls = [tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in pred_ids]

    logger.info(f'Predictions {pred_nls}')
    logger.info(f'Labels {[gold.label_texts for gold in eval_dataset]}')
    if task == 'mrpc' or task == 'qqp':
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for pre, gold in zip(pred_nls, eval_dataset):
            pre = pre.strip().lower()
            gold = gold.label_texts.strip().lower()
            if gold == 'yes' and pre == 'yes':
                tp = tp + 1
            elif gold == 'yes' and pre == 'no':
                fn = fn + 1
            elif gold == 'no' and pre == 'yes':
                fp = fp + 1
            else:
                tn = tn + 1
        precision = tp / (tp + fp) if tp + fp != 0 else 0
        recall = tp / (tp + fn) if tp + fn != 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0
        return f1
    else:
        acc = 0
        for pre, gold in zip(pred_nls, eval_dataset):
            if pre.strip().lower() == gold.label_texts.strip().lower():
                acc += 1
        acc /= len(pred_nls)
        return acc


def create_optimizer_and_scheduler(model, args, num_training_steps):
    model_params = {}
    arch_params = {}
    for n, p in model.named_parameters():
        if n.find("arch_para") != -1:
            arch_params[n] = p
        else:
            model_params[n] = p
    logger.info("***** Model Parameters *****")
    logger.info(model_params.keys())
    logger.info("***** Architecture Parameters *****")
    logger.info(arch_params.keys())
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_model_parameters = [
        {
            "params": [p for n, p in model_params.items() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model_params.items() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer_grouped_arch_parameters = [
        {
            "params": [p for n, p in arch_params.items() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.arch_weight_decay,
        },
        {
            "params": [p for n, p in arch_params.items() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_model_parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
    )
    arch_optimizer = AdamW(
        optimizer_grouped_arch_parameters,
        lr=args.arch_learning_rate,
        betas=(args.arch_adam_beta1, args.arch_adam_beta2),
        eps=args.arch_adam_epsilon,
    )
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_training_steps
    )
    arch_lr_scheduler = get_linear_schedule_with_warmup(
        arch_optimizer, num_warmup_steps=args.arch_warmup_steps, num_training_steps=num_training_steps
    )
    return optimizer, arch_optimizer, lr_scheduler, arch_lr_scheduler


def setup_parallel():
    rank = torch.distributed.get_rank()
    local_rank = os.environ['LOCAL_RANK']
    master_addr = os.environ['MASTER_ADDR']
    master_port = os.environ['MASTER_PORT']
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
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

    setup_parallel()
    set_seed(training_args.seed)

    logger.warning(
        "Process rank: %s, world size: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.world_size,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)
    logger.info("Training/evaluation parameters %s", training_args)


    try:
        num_labels = num_labels_mapping[data_args.task_name]
        output_mode = output_modes_mapping[data_args.task_name]
        logger.info("Task name: {}, number of labels: {}, output mode: {}".format(data_args.task_name, num_labels, output_mode))
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    if retrieval_args.enable_retrieval:
        assert retrieval_args.retrieval_mode != None
        logger.info("Enable Retrievals")
        logger.info("Retrieval parameters %s", retrieval_args)
        retriever = Retriever(
            retrieval_args.retriever_path, 
            retrieval_args.nprobe, 
            topk=retrieval_args.topk,
            device_id=retrieval_args.retriever_device,
        )
        query_encoder = SentenceTransformer(retrieval_args.encoder_path).to(training_args.device)
    else:
        retriever = None
        query_encoder = None

    config = T5Config.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = T5Tokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        use_fast=False,
        cache_dir=model_args.cache_dir,
    )
    if 'prompt' in model_args.few_shot_type:
        if not retrieval_args.enable_retrieval:
            model = T5ForConditionalGeneration.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
            )
        else:
            model = RetrievalAugmentedT5.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
            )

    model_args.transformer_type = transformer_type_mapping[config.model_type]
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

    model.to(training_args.device)
    if training_args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if training_args.do_train:
        # Prepare training data loader
        train_sampler = RandomSampler(train_dataset) if training_args.local_rank == -1 else DistributedSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset, 
            sampler=train_sampler, 
            batch_size=training_args.train_batch_size,
            collate_fn=data_collator_retrieval,
            num_workers=4,
            pin_memory=True,
        )

        num_update_steps_per_epoch = np.max((len(train_dataloader) // training_args.gradient_accumulation_steps, 1))
        if training_args.max_steps > 0:
            t_total = training_args.max_steps
            num_train_epochs = training_args.max_steps // num_update_steps_per_epoch + int(
                training_args.max_steps % num_update_steps_per_epoch > 0
            )
        else:
            t_total = int(len(train_dataloader) // training_args.gradient_accumulation_steps * training_args.num_train_epochs)
            num_train_epochs = training_args.num_train_epochs
        optimizer, arch_optimizer, scheduler, arch_scheduler = create_optimizer_and_scheduler(model, training_args, t_total)

        # Start training
        train_example_num = len(train_dataset)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", train_example_num)
        logger.info("  Num epochs = %d", num_train_epochs)
        logger.info("  Batch size = %d", training_args.train_batch_size)
        logger.info("  Num Batches = %d", math.ceil(train_example_num / training_args.train_batch_size))
        logger.info("  Gradient Accumulation steps = %d", training_args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        best_acc = 0
        model.train()
        model.zero_grad()
        for epoch in range(int(num_train_epochs)):
            if retrieval_args.retrieval_mode == "nas":
                eval_sampler = RandomSampler(eval_dataset) if training_args.local_rank == -1 else DistributedSampler(eval_dataset)
                eval_dataloader = DataLoader(
                    eval_dataset, 
                    sampler=eval_sampler, 
                    batch_size=training_args.eval_batch_size,
                    collate_fn=data_collator_retrieval,
                    num_workers=4,
                    pin_memory=True,
                )
                train_nas(training_args, train_dataloader, eval_dataloader, model, optimizer, scheduler, arch_optimizer, arch_scheduler)
            else:
                train(training_args, retrieval_args, train_dataloader, model, optimizer, scheduler)

            if training_args.do_eval:
                print("Start to evaluation on dev set")
                acc = evaluation(epoch, training_args, retrieval_args, eval_dataset, model, tokenizer, data_args.task_name, 'dev')
                logger.info("  %s = %s "%("acc",str(acc)))
                logger.info("  "+"*"*20) 

                if acc > best_acc:
                    logger.info("  Best acc:%s",acc)
                    logger.info("  "+"*"*20)
                    best_acc = acc

    if training_args.do_predict:
        logger.info("  " + "***** Testing *****")
        logger.info("  Batch size = %d", training_args.eval_batch_size)
        model = model.module if hasattr(model, 'module') else model
        acc = evaluation(-1, training_args, retrieval_args, test_dataset, model, tokenizer, data_args.task_name, 'test')
        logger.info("%s =%s"%("acc", str(acc)))

        if data_args.task_name == "mnli":
            mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
            test_dataset = FewShotDataset(
                mnli_mm_data_args, 
                tokenizer=tokenizer, 
                mode="test", 
                transformer_type=model_args.transformer_type, 
                retriever=retriever, 
                query_encoder=query_encoder
            )
            acc = evaluation(-1, training_args, retrieval_args, test_dataset, model, tokenizer, data_args.task_name, 'test')
            logger.info("%s =%s"%("acc", str(acc)))


if __name__ == "__main__":
    main()
