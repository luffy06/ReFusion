import logging
logger = logging.getLogger(__name__)

from transformers.models.gemma.modeling_gemma import (
    GemmaForCausalLM
)
from models.layers import RetrievalWrapper

class RetrievalAugmentedGemma(GemmaForCausalLM, RetrievalWrapper):

    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        input_ids=None,
        input_texts=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        neighbors=None,
        neighbor_texts=None,
        return_dict=None,
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
            position_ids,
            past_key_values,
            inputs_embeds,
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
