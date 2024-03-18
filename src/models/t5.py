import logging
logger = logging.getLogger(__name__)

from transformers.models.t5.modeling_t5 import (
    T5ForConditionalGeneration
)
from models.layers import RetrievalWrapper

class RetrievalAugmentedT5(T5ForConditionalGeneration, RetrievalWrapper):

    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        input_ids=None,
        input_texts=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        neighbors=None,
        neighbor_texts=None,
        **kwargs,
    ):
        if self.enable_retrieval and self.retrieve_texts and neighbors != None:
            # query_emb = self.query_encoder.encode(input_texts)
            # neighbors, neighbor_texts = self.retriever.search(query_emb, return_texts=True)
            # texts = '\n'.join(neighbor_texts[0])
            self.retriever.save_in_cache(neighbors)

        return super().forward(
            input_ids,
            attention_mask,
            decoder_input_ids,
            decoder_attention_mask,
            head_mask,
            decoder_head_mask,
            cross_attn_head_mask,
            encoder_outputs,
            past_key_values,
            inputs_embeds,
            decoder_inputs_embeds,
            labels,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
        )

    def generate(
        self,
        input_ids,
        input_texts=None,
        attention_mask=None,
        generation_config=None,
        logits_processor=None,
        stopping_criteria=None,
        prefix_allowed_tokens_fn=None,
        synced_gpus=None,
        assistant_model=None,
        streamer=None,
        negative_prompt_ids=None,
        negative_prompt_attention_mask=None,
        neighbors=None,
        neighbor_texts=None,
        max_length=None,
        **kwargs,
    ):
        if self.enable_retrieval and self.retrieve_texts and neighbors != None:
            # query_emb = self.query_encoder.encode(input_texts)
            # neighbors, neighbor_texts = self.retriever.search(query_emb, return_texts=True)
            # texts = '\n'.join(neighbor_texts[0])
            self.retriever.save_in_cache(neighbors)

        return super().generate(
            input_ids,
            generation_config,
            logits_processor,
            stopping_criteria,
            prefix_allowed_tokens_fn,
            synced_gpus,
            assistant_model,
            streamer,
            negative_prompt_ids,
            negative_prompt_attention_mask,
            attention_mask=attention_mask,
            max_length=max_length,
            neighbors=neighbors,
            neighbor_texts=neighbor_texts,
            **kwargs
        )
