import torch
import torch.nn as nn
import logging
logger = logging.getLogger(__name__)

from transformers.models.roberta.modeling_roberta import (
    RobertaPreTrainedModel,
    RobertaModel, 
    RobertaLMHead, 
    RobertaClassificationHead
)
from models.layers import RetrievalWrapper

class RobertaForPromptFinetuning(RobertaPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.classifier = RobertaClassificationHead(config)
        self.lm_head = RobertaLMHead(config)
        self.init_weights()

        # These attributes should be assigned once the model is initialized
        self.label_word_list = None

        # For regression
        self.lb = None
        self.ub = None

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        mask_pos=None,
    ):
        batch_size = input_ids.size(0)

        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()

        # Encode everything
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Get <mask> token representation
        sequence_output, pooled_output = outputs[:2]
        sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos]

        # Logits over vocabulary tokens
        prediction_mask_scores = self.lm_head(sequence_mask_output)

        # Return logits for each label
        logits = []
        for label_id in range(len(self.label_word_list)):
            logits.append(prediction_mask_scores[:, self.label_word_list[label_id]].unsqueeze(-1))
        logits = torch.cat(logits, -1)

        # Regression task
        if self.num_labels == 1:
            logsoftmax = nn.LogSoftmax(-1)
            logits = logsoftmax(logits) # Log prob of right polarity

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Regression task
                loss_fct = nn.KLDivLoss(log_target=True)
                labels = torch.stack([1 - (labels.view(-1) - self.lb) / (self.ub - self.lb), (labels.view(-1) - self.lb) / (self.ub - self.lb)], -1)
                loss = loss_fct(logits.view(-1, 2), labels)
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        output = (logits,)
        if self.num_labels == 1:
            # Regression output
            output = (torch.exp(logits[..., 1].unsqueeze(-1)) * (self.ub - self.lb) + self.lb,)
        return ((loss,) + output) if loss is not None else output


class RetrievalAugmentedRoberta(RobertaForPromptFinetuning, RetrievalWrapper):
    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        input_ids=None,
        input_texts=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        mask_pos=None,
        neighbors=None,
        neighbor_texts=None,
    ):
        if self.enable_retrieval and self.retrieve_texts:
            # For online retrieval
            # query_emb = self.query_encoder.encode(input_texts, show_progress_bar=False)
            # neighbors, neighbor_texts = self.retriever.search(query_emb, return_texts=True)
            self.retriever.save_in_cache(neighbors)
        
        return super().forward(
            input_ids, 
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            encoder_hidden_states,
            encoder_attention_mask,
            labels,
            past_key_values,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
            mask_pos,
        )

