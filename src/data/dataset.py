"""Dataset utils for different data settings for GLUE."""
import os
import logging
import torch
import time
import json
import dataclasses
import numpy as np
import pandas as pd
from filelock import FileLock
from data.processors import (
    processors_mapping, 
    num_labels_mapping, 
    output_modes_mapping, 
    compute_metrics_mapping, 
    median_mapping
)
from transformers.data.processors.utils import InputFeatures
from dataclasses import dataclass
from typing import List, Optional, Union, Dict, Mapping, Any
from copy import deepcopy
from tqdm import tqdm

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class EncoderOnlyInputFeatures(InputFeatures):
    """
    Inherit from Transformers' InputFeatuers.
    """

    input_ids: List[int]
    input_texts: Optional[List[str]] = None
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    mask_pos: Optional[List[int]] = None # Position of the mask token
    neighbors: np.array = None
    neighbor_texts: Optional[List[str]] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"

@dataclass(frozen=True)
class EncoderDecoderInputFeatures(InputFeatures):
    """
    Inherit from Transformers' InputFeatuers.
    """

    input_ids: List[int]
    input_texts: Optional[List[str]] = None
    attention_mask: Optional[List[int]] = None
    labels: Optional[List[int]] = None
    label_texts: Optional[List[str]] = None
    decoder_attention_mask: Optional[List[int]] = None
    neighbors: np.array = None
    neighbor_texts: Optional[List[str]] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"

@dataclass(frozen=True)
class DecoderOnlyInputFeatures(InputFeatures):
    """
    Inherit from Transformers' InputFeatuers.
    """

    input_ids: List[int]
    input_texts: Optional[List[str]] = None
    attention_mask: Optional[List[int]] = None
    labels: Optional[List[int]] = None
    label_texts: Optional[List[str]] = None
    neighbors: np.array = None
    neighbor_texts: Optional[List[str]] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"

def input_example_to_string(example, sep_token): 
    if example.text_b is None:
        return example.text_a
    else:
        # Warning: very simple hack here
        return example.text_a + ' ' + sep_token + ' ' + example.text_b

def input_example_to_tuple(example): 
    if example.text_b is None:
        if pd.isna(example.text_a) or example.text_a is None:
            return ['']
            logger.warn("Empty input")
        else:
            return [example.text_a]
    else:
        return [example.text_a, example.text_b]

def tokenize_input_encoder_only(
    input_text_list, 
    label_id, 
    max_length, 
    tokenizer, 
    task_name=None, 
    prompt=False, 
    template=None,
    label_word_list=None, 
    first_sent_limit=None,
    other_sent_limit=None,
    truncate_head=False,
    ref_text_list=None,
    return_texts=False
):
    def enc(text):
        return tokenizer.encode(text, add_special_tokens=False)

    def dec(ids):
        return tokenizer.decode(ids)

    input_ids = []
    attention_mask = []
    token_type_ids = [] # Only for BERT
    mask_pos = None # Position of the mask token
    label_ids = []

    if prompt:
        """
        Concatenate all sentences and prompts based on the provided template.
        Template example: '*cls*It was*mask*.*sent_0**<sep>*label_0:*sent_1**<sep>**label_1*:*sent_2**<sep>*'
        *xx* represent variables:
            *cls*: cls_token
            *mask*: mask_token
            *sep*: sep_token
            *sep+*: sep_token, also means +1 for segment id
            *sent_i*: sentence i (input_text_list[i])
            *sent-_i*: same as above, but delete the last token
            *sentl_i*: same as above, but use lower case for the first word
            *sentl-_i*: same as above, but use lower case for the first word and delete the last token
            *+sent_i*: same as above, but add a space before the sentence
            *+sentl_i*: same as above, but add a space before the sentence and use lower case for the first word
            *ref_i*: retrieval i (ref_text_list[i])

        Use "_" to replace space.
        PAY ATTENTION TO SPACE!! DO NOT leave space before variables, for this will lead to extra space token.
        """
        assert template is not None

        special_token_mapping = ['cls', 'mask', 'sep', 'sep+']
        template_list = template.split('*') # Get variable list in the template
        segment_id = 0 # Current segment id. Segment id +1 if encountering sep+.

        for part_id, part in enumerate(template_list):
            new_tokens = []
            segment_plus_1_flag = False
            if part in special_token_mapping:
                if part == 'cls':
                    new_tokens += [tokenizer.cls_token_id]
                elif part == 'mask':
                    new_tokens += [tokenizer.mask_token_id]
                elif part == 'sep' or part == 'sep+':
                    new_tokens += [tokenizer.sep_token_id]
                else:
                    raise NotImplementedError(f'Unrecognized special token {part}')
                if part == 'sep+':
                    segment_plus_1_flag = True
            elif part[:5] == 'sent_':
                sent_id = int(part.split('_')[1])
                new_tokens += enc(input_text_list[sent_id]) 
            elif part[:6] == '+sent_':
                # Add space
                sent_id = int(part.split('_')[1])
                new_tokens += enc(' ' + input_text_list[sent_id])
            elif part[:6] == 'sent-_':
                # Delete the last token
                sent_id = int(part.split('_')[1])
                new_tokens += enc(input_text_list[sent_id][:-1])
            elif part[:6] == 'sentl_':
                # Lower case the first token
                sent_id = int(part.split('_')[1])
                text = input_text_list[sent_id]
                text = text[:1].lower() + text[1:]
                new_tokens += enc(text)
            elif part[:7] == '+sentl_':
                # Lower case the first token and add space 
                sent_id = int(part.split('_')[1])
                text = input_text_list[sent_id]
                text = text[:1].lower() + text[1:]
                new_tokens += enc(' ' + text)
            elif part[:7] == 'sentl-_':
                # Lower case the first token and discard the last token
                sent_id = int(part.split('_')[1])
                text = input_text_list[sent_id]
                text = text[:1].lower() + text[1:]
                new_tokens += enc(text[:-1])
            elif part[:6] == 'sentu_':
                # Upper case the first token
                sent_id = int(part.split('_')[1])
                text = input_text_list[sent_id]
                text = text[:1].upper() + text[1:]
                new_tokens += enc(text)
            elif part[:7] == '+sentu_':
                # Upper case the first token and add space
                sent_id = int(part.split('_')[1])
                text = input_text_list[sent_id]
                text = text[:1].upper() + text[1:]
                new_tokens += enc(' ' + text)
            elif part[:4] == 'ref_':
                if ref_text_list != None:
                    ref_id = int(part.split('_')[1])
                    ref_text_list[ref_id].reverse()
                    new_tokens += enc(' '.join(ref_text_list[ref_id]))
            else:
                # Just natural language prompt
                part = part.replace('_', ' ') 
                # handle special case when T5 tokenizer might add an extra space
                if len(part) == 1:
                    new_tokens.append(tokenizer._convert_token_to_id(part))
                else:
                    new_tokens += enc(part)

            if part[:4] == 'sent' or part[1:5] == 'sent':
                # If this part is the sentence, limit the sentence length
                sent_id = int(part.split('_')[1])
                if sent_id == 0:
                    if first_sent_limit is not None:
                        new_tokens = new_tokens[:first_sent_limit]
                else:
                    if other_sent_limit is not None:
                        new_tokens = new_tokens[:other_sent_limit]

            input_ids += new_tokens
            attention_mask += [1 for i in range(len(new_tokens))]
            token_type_ids += [segment_id for i in range(len(new_tokens))]

            if segment_plus_1_flag:
                segment_id += 1
    
        label_ids += [label_id]
    else:
        input_ids += [tokenizer.cls_token_id] 
        attention_mask += [1]
        token_type_ids += [0]

        for sent_id, input_text in enumerate(input_text_list):
            if input_text is None:
                # Do not have text_b
                continue
            if pd.isna(input_text) or input_text is None:
                # Empty input
                input_text = ''
            input_tokens = enc(input_text) + [tokenizer.sep_token_id]
            input_ids += input_tokens
            attention_mask += [1 for i in range(len(input_tokens))]
            token_type_ids += [sent_id for i in range(len(input_tokens))]

        label_ids += [label_id]

    # Padding
    if len(input_ids) > max_length:
        # If using sentence limit, the total length still exceeds the maximum limit, report a warning
        logger.warn("Input exceeds max_length limit: {}".format(len(input_ids)))

    while len(input_ids) < max_length:
        input_ids.append(tokenizer.pad_token_id)
        attention_mask.append(0)
        token_type_ids.append(0)

    # Truncate
    if len(input_ids) > max_length:
        if truncate_head:
            input_ids = input_ids[-(max_length-1):]
            attention_mask = attention_mask[-(max_length-1):]
            token_type_ids = token_type_ids[-(max_length-1):]
            input_ids = [tokenizer.cls_token_id] + input_ids
            attention_mask = [1] + attention_mask
            token_type_ids = [0] + token_type_ids
        else:
            # Default is to truncate the tail
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            token_type_ids = token_type_ids[:max_length]

    # Find mask token
    if prompt:
        mask_pos = [input_ids.index(tokenizer.mask_token_id)]
        # Make sure that the masked position is inside the max_length
        assert mask_pos[0] < max_length

    result = {'input_ids': input_ids, 'attention_mask': attention_mask}
    if 'BERT' in type(tokenizer).__name__:
        # Only provide token type ids for BERT
        result['token_type_ids'] = token_type_ids

    if prompt:
        result['mask_pos'] = mask_pos
    
    if return_texts:
        result['input_texts'] = ' '.join(input_text_list)

    return result

def tokenize_input_encoder_decoder(
    input_text_list, 
    label_id, 
    max_length, 
    tokenizer, 
    task_name=None, 
    prompt=False, 
    template=None,
    label_word_list=None, 
    first_sent_limit=None,
    other_sent_limit=None,
    truncate_head=False,
    ref_text_list=None,
    return_texts=False
):
    def enc(text, max_length=-1):
        return tokenizer.encode(
            text, 
            max_length=max_length if max_length > 0 else None,
            padding='max_length',
            add_special_tokens=False, 
            truncation=True if max_length > 0 else False,
        )

    def dec(ids):
        return tokenizer.decode(ids, skip_special_tokens=True)

    input_ids = []
    attention_mask = []
    label_ids = []
    label_mask = []
    source_text = ''
    target_text = ''

    if prompt:
        """
        Concatenate all sentences and prompts based on the provided template.
        Template example:'*sent_0* It was*<extra_id_0>*.*sep**<extra_id_0>**label**<extra_id_1>*'
        *xx* represent variables:
            *<extra_id_i>*: the mask token in T5
            *sent_i*: sentence i (input_text_list[i])
            *sent-_i*: same as above, but delete the last token
            *sentl_i*: same as above, but use lower case for the first word
            *sentl-_i*: same as above, but use lower case for the first word and delete the last token
            *+sent_i*: same as above, but add a space before the sentence
            *+sentl_i*: same as above, but add a space before the sentence and use lower case for the first word
            *ref_i*: retrieval i (ref_text_list[i])
            *ref-_i*: same as above, but delete the last token
            *refl_i*: same as abovez, but use lower case for the first word
            *refl-_i*: same as above, but use lower case for the first word and delete the last token
            *+ref_i*: same as above, but add a space before the retrieval
            *+refl_i*: same as above, but add a space before the retrieval and use lower case for the first word

        Use "_" to replace space.
        PAY ATTENTION TO SPACE!! DO NOT leave space before variables, for this will lead to extra space token.
        """
        assert template is not None
        
        template_list = template.split('*') # Get variable list in the template
        gen_input_ids = True

        for part_id, part in enumerate(template_list):
            new_texts = ''
            if part.startswith('<extra_id_'):
                new_texts += part
            elif part == 'sep':
                gen_input_ids = False
            elif part == 'eos':
                new_texts += tokenizer.eos_token
            elif part == 'label':
                new_texts += dec(label_id)
            elif part[:5] == 'sent_':
                sent_id = int(part.split('_')[1])
                new_texts += input_text_list[sent_id]
            elif part[:6] == '+sent_':
                # Add space
                sent_id = int(part.split('_')[1])
                new_texts += ' ' + input_text_list[sent_id]
            elif part[:6] == 'sent-_':
                # Delete the last token
                sent_id = int(part.split('_')[1])
                new_texts += input_text_list[sent_id][:-1]
            elif part[:6] == 'sentl_':
                # Lower case the first token
                sent_id = int(part.split('_')[1])
                text = input_text_list[sent_id]
                text = text[:1].lower() + text[1:]
                new_texts += text
            elif part[:7] == '+sentl_':
                # Lower case the first token and add space 
                sent_id = int(part.split('_')[1])
                text = input_text_list[sent_id]
                text = text[:1].lower() + text[1:]
                new_texts += ' ' + text
            elif part[:7] == 'sentl-_':
                # Lower case the first token and discard the last token
                sent_id = int(part.split('_')[1])
                text = input_text_list[sent_id]
                text = text[:1].lower() + text[1:]
                new_texts += text[:-1]
            elif part[:6] == 'sentu_':
                # Upper case the first token
                sent_id = int(part.split('_')[1])
                text = input_text_list[sent_id]
                text = text[:1].upper() + text[1:]
                new_texts += text
            elif part[:7] == '+sentu_':
                # Upper case the first token and add space
                sent_id = int(part.split('_')[1])
                text = input_text_list[sent_id]
                text = text[:1].upper() + text[1:]
                new_texts += ' ' + text
            elif part[:4] == 'ref_':
                ref_id = int(part.split('_')[1])
                new_texts += ' '.join(ref_text_list[ref_id])
            else:
                # Just natural language prompt
                part = part.replace('_', ' ') 
                new_texts += part

            if part[:4] == 'sent' or part[1:5] == 'sent':
                # If this part is the sentence, limit the sentence length
                sent_id = int(part.split('_')[1])
                if sent_id == 0:
                    if first_sent_limit is not None:
                        new_texts = new_texts[:first_sent_limit]
                else:
                    if other_sent_limit is not None:
                        new_texts = new_texts[:other_sent_limit]

            if gen_input_ids:
                source_text += new_texts
            else:
                target_text += new_texts
    else:
        """
        Treat the problem as a translation problem.
            The input sentences are source texts;
            The labels are target texts;
            We use the encoder-decoder models to learn the translation 
            between source texts and target texts.
        """
        source_text = ' '.join(input_text_list)
        target_text = dec(label_id)

    input_ids = enc(source_text, max_length=max_length)
    label_ids = enc(target_text)
    attention_mask += [1 if input_ids[i] != tokenizer.pad_token_id else 0 for i in range(len(input_ids))]
    label_mask += [1 if label_ids[i] != tokenizer.pad_token_id else 0 for i in range(len(label_ids))]

    # Padding
    if first_sent_limit is not None and len(input_ids) > max_length:
        # If using sentence limit, the total length still exceeds the maximum limit, report a warning
        logger.warn("Input exceeds max_length limit: {}".format(tokenizer.decode(input_ids)))


    result = {
        'input_ids': input_ids, 
        'attention_mask': attention_mask, 
        'labels': label_ids,
        'decoder_attention_mask': label_mask,
    }

    if return_texts:
        result['input_texts'] = ' '.join(input_text_list)
        result['label_texts'] = dec(label_ids)

    return result

def tokenize_input_decoder_only(
    input_text_list, 
    label_id, 
    max_length, 
    tokenizer, 
    task_name=None, 
    prompt=False, 
    template=None,
    label_word_list=None, 
    first_sent_limit=None,
    other_sent_limit=None,
    truncate_head=False,
    ref_text_list=None,
    return_texts=False
):
    def enc(text, max_length=-1):
        return tokenizer.encode(
            text, 
            max_length=max_length if max_length > 0 else None,
            padding='max_length',
            add_special_tokens=False, 
            truncation=True if max_length > 0 else False,
        )

    def dec(ids):
        return tokenizer.decode([ids] if type(ids) is int else ids, skip_special_tokens=True)

    input_ids = []
    attention_mask = []
    label_ids = []
    source_text = ''
    target_text = ''

    if prompt:
        """
        Concatenate all sentences and prompts based on the provided template.
        Template example:'*sent_0* It was*<extra_id_0>*.*sep**<extra_id_0>**label**<extra_id_1>*'
        *xx* represent variables:
            *<extra_id_i>*: the mask token in T5
            *sent_i*: sentence i (input_text_list[i])
            *sent-_i*: same as above, but delete the last token
            *sentl_i*: same as above, but use lower case for the first word
            *sentl-_i*: same as above, but use lower case for the first word and delete the last token
            *+sent_i*: same as above, but add a space before the sentence
            *+sentl_i*: same as above, but add a space before the sentence and use lower case for the first word
            *ref_i*: retrieval i (ref_text_list[i])
            *ref-_i*: same as above, but delete the last token
            *refl_i*: same as abovez, but use lower case for the first word
            *refl-_i*: same as above, but use lower case for the first word and delete the last token
            *+ref_i*: same as above, but add a space before the retrieval
            *+refl_i*: same as above, but add a space before the retrieval and use lower case for the first word

        Use "_" to replace space.
        PAY ATTENTION TO SPACE!! DO NOT leave space before variables, for this will lead to extra space token.
        """
        assert template is not None
        
        template_list = template.split('*') # Get variable list in the template
        gen_input_ids = True

        for part_id, part in enumerate(template_list):
            new_texts = ''
            if part.startswith('<extra_id_'):
                new_texts += part
            elif part == 'sep':
                gen_input_ids = False
            elif part == 'eos':
                new_texts += tokenizer.eos_token
            elif part == 'label':
                new_texts += dec(label_id)
            elif part[:5] == 'sent_':
                sent_id = int(part.split('_')[1])
                new_texts += input_text_list[sent_id]
            elif part[:6] == '+sent_':
                # Add space
                sent_id = int(part.split('_')[1])
                new_texts += ' ' + input_text_list[sent_id]
            elif part[:6] == 'sent-_':
                # Delete the last token
                sent_id = int(part.split('_')[1])
                new_texts += input_text_list[sent_id][:-1]
            elif part[:6] == 'sentl_':
                # Lower case the first token
                sent_id = int(part.split('_')[1])
                text = input_text_list[sent_id]
                text = text[:1].lower() + text[1:]
                new_texts += text
            elif part[:7] == '+sentl_':
                # Lower case the first token and add space 
                sent_id = int(part.split('_')[1])
                text = input_text_list[sent_id]
                text = text[:1].lower() + text[1:]
                new_texts += ' ' + text
            elif part[:7] == 'sentl-_':
                # Lower case the first token and discard the last token
                sent_id = int(part.split('_')[1])
                text = input_text_list[sent_id]
                text = text[:1].lower() + text[1:]
                new_texts += text[:-1]
            elif part[:6] == 'sentu_':
                # Upper case the first token
                sent_id = int(part.split('_')[1])
                text = input_text_list[sent_id]
                text = text[:1].upper() + text[1:]
                new_texts += text
            elif part[:7] == '+sentu_':
                # Upper case the first token and add space
                sent_id = int(part.split('_')[1])
                text = input_text_list[sent_id]
                text = text[:1].upper() + text[1:]
                new_texts += ' ' + text
            elif part[:4] == 'ref_':
                ref_id = int(part.split('_')[1])
                new_texts += ' '.join(ref_text_list[ref_id])
            else:
                # Just natural language prompt
                part = part.replace('_', ' ') 
                new_texts += part

            if part[:4] == 'sent' or part[1:5] == 'sent':
                # If this part is the sentence, limit the sentence length
                sent_id = int(part.split('_')[1])
                if sent_id == 0:
                    if first_sent_limit is not None:
                        new_texts = new_texts[:first_sent_limit]
                else:
                    if other_sent_limit is not None:
                        new_texts = new_texts[:other_sent_limit]

            if gen_input_ids:
                source_text += new_texts
            else:
                target_text += new_texts
    else:
        """
        Treat the problem as a translation problem.
            The input sentences are source texts;
            The labels are target texts;
            We use the encoder-decoder models to learn the translation 
            between source texts and target texts.
        """
        source_text = ' '.join(input_text_list)
        target_text = dec(label_id)

    input_ids = enc(source_text, max_length=max_length)
    label_ids = enc(target_text)
    attention_mask += [1 if input_ids[i] != tokenizer.pad_token_id else 0 for i in range(len(input_ids))]

    # Padding
    if first_sent_limit is not None and len(input_ids) > max_length:
        # If using sentence limit, the total length still exceeds the maximum limit, report a warning
        logger.warn("Input exceeds max_length limit: {}".format(tokenizer.decode(input_ids)))


    result = {
        'input_ids': input_ids, 
        'attention_mask': attention_mask, 
        'labels': label_ids,
    }

    if return_texts:
        result['input_texts'] = ' '.join(input_text_list)
        result['label_texts'] = dec(label_ids)

    return result

def data_collator_retrieval(features: List[Any]) -> Dict[str, Any]:
    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]
    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)
    if "input_texts" in first and first["input_texts"] is not None:
        batch["input_texts"] = [f["input_texts"] for f in features]
    if "neighbors" in first and first["neighbors"] is not None:
        batch["neighbors"] = torch.tensor(np.stack([f["neighbors"] for f in features]))
    if "neighbor_texts" in first and first["neighbor_texts"] is not None:
        batch["neighbor_texts"] = []
        for f in features:
            for text in f["neighbor_texts"]:
                batch['neighbor_texts'].append(text)
    
    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ("label", "label_ids", "input_texts", "neighbors", "neighbor_texts") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.tensor(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.tensor([f[k] for f in features])
    return batch


class FewShotDataset(torch.utils.data.Dataset):
    """Few-shot dataset."""

    def __init__(self, 
        args, 
        tokenizer, 
        cache_dir=None, 
        mode="train", 
        transformer_type="encoder-only",
        retriever=None,
        query_encoder=None,
    ):
        self.args = args
        self.task_name = args.task_name
        self.processor = processors_mapping[args.task_name]
        self.tokenizer = tokenizer
        self.mode = mode
        self.transformer_type = transformer_type

        assert mode in ["train", "dev", "test"]

        # Get label list and (for prompt) label word list
        self.label_list = self.processor.get_labels()
        self.num_labels = len(self.label_list)
        if args.prompt:
            assert args.mapping is not None
            self.label_to_word = eval(args.mapping)

            for key in self.label_to_word:
                # For RoBERTa/BART/T5, tokenization also considers space, so we use space+word as label words.
                if self.label_to_word[key][0] not in ['<', '[', '.', ',']:
                    # Make sure space+word is in the vocabulary
                    assert len(tokenizer.tokenize(' ' + self.label_to_word[key])) == 1
                    self.label_to_word[key] = tokenizer._convert_token_to_id(tokenizer.tokenize(' ' + self.label_to_word[key])[0])
                else:
                    self.label_to_word[key] = tokenizer._convert_token_to_id(self.label_to_word[key])
                logger.info("Label {} to word {} ({})".format(key, tokenizer._convert_id_to_token(self.label_to_word[key]), self.label_to_word[key]))
            
            if len(self.label_list) > 1:
                self.label_word_list = [self.label_to_word[label] for label in self.label_list]
            else:
                # Regression task
                # '0' represents low polarity and '1' represents high polarity.
                self.label_word_list = [self.label_to_word[label] for label in ['0', '1']]
        else:
            self.label_to_word = None
            self.label_word_list = None

        # Load cache
        # Cache name distinguishes mode, task name, tokenizer, and length. So if you change anything beyond these elements, make sure to clear your cache.
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else args.data_dir,
            "cached_{}_{}_{}_{}".format(
                mode,
                tokenizer.__class__.__name__,
                str(args.max_seq_length),
                args.task_name,
            ),
        )

        logger.info(f"Creating/loading examples from dataset file at {args.data_dir}")

        logger.info(f"Creating features from dataset file at {args.data_dir}")

        if mode == "dev":
            self.query_examples = self.processor.get_dev_examples(args.data_dir)
        elif mode == "test":
            self.query_examples = self.processor.get_test_examples(args.data_dir)
        else:
            self.query_examples = self.processor.get_train_examples(args.data_dir)

        self.num_sample = 1
        # Size is expanded by num_sample
        self.size = len(self.query_examples)

        all_neighbor_embs = []
        all_neighbor_texts = []
        if retriever != None:
            step = 2 if self.query_examples[0].text_b != None else 1
            batch_size = 32
            if len(self.query_examples) % batch_size == 0:
                num_batches = int(len(self.query_examples) / batch_size)
            else:
                num_batches = int(len(self.query_examples) // batch_size) + 1
            for i in tqdm(range(num_batches)):
                l = i * batch_size
                r = np.min(((i + 1) * batch_size, len(self.query_examples)))
                batch_input_texts = []
                for j in range(l, r):
                    input_text_list = input_example_to_tuple(self.query_examples[j])
                    batch_input_texts += input_text_list
                
                query_emb = query_encoder.encode(batch_input_texts, show_progress_bar=False)
                batch_neighbors = retriever.search(query_emb)
                for j in range(r - l):
                    embs = []
                    texts = []
                    for k in range(step):
                        embs.append(batch_neighbors[j*step+k]['emb'])
                        texts = texts + batch_neighbors[j*step+k]['text']
                    embs = np.concatenate(embs, axis=1)
                    all_neighbor_embs.append(embs)
                    all_neighbor_texts.append(texts)
            all_neighbor_embs = np.concatenate(all_neighbor_embs, axis=0)

        self.features = []
        for i, example in enumerate(self.query_examples):
            self.features.append(self.convert_fn(
                example=example,
                label_list=self.label_list,
                prompt=args.prompt,
                template=args.template,
                label_word_list=self.label_word_list,
                verbose=True if i == 0 else False,
                neighbor_embs=all_neighbor_embs[i] if len(all_neighbor_embs) else None,
                neighbor_texts=all_neighbor_texts[i] if len(all_neighbor_embs) else None,
            ))

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        return self.features[i]

    def get_labels(self):
        return self.label_list

    def convert_fn(
        self,
        example,
        label_list=None,
        prompt=False,
        template=None,
        label_word_list=None,
        verbose=False,
        neighbor_embs=None,
        neighbor_texts=None,
    ):
        """
        Returns a list of processed "InputFeatures".
        """
        max_length = self.args.max_seq_length    

        # Prepare labels
        label_map = {label: i for i, label in enumerate(label_list)} # Mapping the label names to label ids
        if len(label_list) == 1:
            # Regression
            label_map = {'0': 0, '1': 1}

        # Get example's label id (for training/inference)
        if example.label is None:
            example_label = None
        elif len(label_list) == 1:
            # Regerssion
            example_label = float(example.label)
            label_word_id = None
        else:
            example_label = label_map[example.label]
            label_word_id = None if example_label == None else label_word_list[example_label]

        input_text_list = input_example_to_tuple(example)
        # if self.retriever != None:
        #     query_emb = self.query_encoder.encode(input_text_list, show_progress_bar=False)
        #     neighbors, neighbor_texts = self.retriever.search(query_emb, return_texts=True)
        # else:
        #     neighbors, neighbor_texts = None, None

        if self.transformer_type == "encoder-only":
            inputs = tokenize_input_encoder_only(
                input_text_list=input_text_list,
                label_id=label_word_id,
                max_length=max_length,
                tokenizer=self.tokenizer,
                task_name=self.args.task_name,
                prompt=prompt,
                template=template,
                label_word_list=label_word_list,
                first_sent_limit=self.args.first_sent_limit,
                other_sent_limit=self.args.other_sent_limit,
                truncate_head=self.args.truncate_head,
                ref_text_list=neighbor_texts,
                return_texts=self.args.return_texts,
            )
            features = EncoderOnlyInputFeatures(
                **inputs, 
                label=example_label, 
                neighbors=neighbor_embs,
                neighbor_texts=neighbor_texts,
            )
        elif self.transformer_type == "encoder-decoder":
            inputs = tokenize_input_encoder_decoder(
                input_text_list=input_text_list,
                label_id=label_word_id,
                max_length=max_length,
                tokenizer=self.tokenizer,
                task_name=self.args.task_name,
                prompt=prompt,
                template=template,
                label_word_list=label_word_list,
                first_sent_limit=self.args.first_sent_limit,
                other_sent_limit=self.args.other_sent_limit,
                truncate_head=self.args.truncate_head,
                ref_text_list=neighbor_texts,
                return_texts=self.args.return_texts,
            )
            features = EncoderDecoderInputFeatures(
                **inputs,
                neighbors=neighbor_embs,
                neighbor_texts=neighbor_texts,
            )
        elif self.transformer_type == "decoder-only":
            inputs = tokenize_input_decoder_only(
                input_text_list=input_text_list,
                label_id=label_word_id,
                max_length=max_length,
                tokenizer=self.tokenizer,
                task_name=self.args.task_name,
                prompt=prompt,
                template=template,
                label_word_list=label_word_list,
                first_sent_limit=self.args.first_sent_limit,
                other_sent_limit=self.args.other_sent_limit,
                truncate_head=self.args.truncate_head,
                ref_text_list=neighbor_texts,
                return_texts=self.args.return_texts,
            )
            features = DecoderOnlyInputFeatures(
                **inputs,
                neighbors=neighbor_embs,
                neighbor_texts=neighbor_texts,
            )
        else:
            raise NotImplementedError(f'Unsupported transformer type {self.transformer_type}')

        if verbose:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input ids: %s" % features.input_ids)
            logger.info("attention mask: %s" % features.attention_mask)
            logger.info("label: %s" % example.label)
            logger.info("input texts: \n%s" % self.tokenizer.decode(features.input_ids))
            if neighbor_texts != None:
                logger.info("neighbors of 1st sentence: \n%s" % '\n'.join([f'Top-{i+1}: {text}'for i, text in enumerate(neighbor_texts[:len(neighbor_texts)//2])]))
                if len(neighbor_texts) > 1:
                    logger.info("neighbors of 2nd sentence: \n%s" % '\n'.join([f'Top-{i+1}: {text}'for i, text in enumerate(neighbor_texts[len(neighbor_texts)//2:])]))
            else:
                logger.info("No neighbors")

        return features
