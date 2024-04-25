import os
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessorList,
    PreTrainedModel,
    PreTrainedTokenizer,
    SequenceBiasLogitsProcessor,
    StoppingCriteria,
    StoppingCriteriaList,
)
from transformers import set_seed as hf_set_seed

from src.decoders.model_backend import ModelBackend, ModelOutput, TokenScore

HF_CACHE_DIR = os.environ.get("HF_CACHE_DIR")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HFModel(ModelBackend):
    """
    Wrapper around HF Transformers model inference
    """

    def __init__(self, hf_model: PreTrainedModel, hf_tokenizer: PreTrainedTokenizer):
        self.model = hf_model
        self.tokenizer = hf_tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def get_tokenizer(self):
        return self.tokenizer

    def generate(
        self,
        prompts: list[str],
        n: int = 1,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: Optional[int] = None,
        max_tokens: int = 128,
        use_beam_search: bool = False,
        length_penalty: float = 1.0,
        early_stopping: bool = False,
        stop: Optional[list[str]] = None,  # TODO: make this work
        logprobs: int = 0,
        logit_bias: Optional[dict[int, float]] = None,
        seed: Optional[int] = None,
        ### HF specific parameters
        repetition_penalty: float = 1.0,
        num_beam_groups: int = 1,
        diversity_penalty: float = 0.0,
    ) -> ModelOutput:
        inputs = self.tokenizer(prompts, return_tensors="pt").to(self.model.device)

        if seed is not None:
            hf_set_seed(seed)

        prefix_lengths = inputs.attention_mask.sum(dim=-1)
        prefix_lengths = [[int(length.detach().cpu())] * n for length in prefix_lengths]
        prefix_lengths = [n for lengths in prefix_lengths for n in lengths]

        lps = LogitsProcessorList()
        if logit_bias is not None:
            lps.append(
                SequenceBiasLogitsProcessor({(k,): v for k, v in logit_bias.items()})
            )

        stopping_criteria = StoppingCriteriaList()
        if stop is not None:
            tokenizer = self.get_tokenizer()
            for stop_str in stop:
                stopping_criteria.append(StopAtStr(stop_str, tokenizer, keep_word=True))

        generation_args = {
            "num_return_sequences": n,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_new_tokens": max_tokens,
            "do_sample": not use_beam_search,
            "length_penalty": length_penalty,
            "early_stopping": early_stopping,
            "num_beam_groups": num_beam_groups,
            "diversity_penalty": diversity_penalty,
            "output_scores": logprobs > 0,
            "return_dict_in_generate": True,
            "logits_processor": lps,
            "stopping_criteria": stopping_criteria,
            "repetition_penalty": repetition_penalty,
            "renormalize_logits": True,  # recommended to set true
            "output_attentions": False,
            "output_hidden_states": False,
        }
        output = self.model.generate(**inputs, **generation_args)

        tokenizer = self.get_tokenizer()

        output.past_key_values = None
        texts = []
        for prefix_length, seq in zip(prefix_lengths, output.sequences):
            texts.append(
                self.tokenizer.decode(seq[prefix_length:], skip_special_tokens=True)
            )

        token_logprobs = []
        if output.scores is not None:
            token_logprobs = self._get_logprobs(
                output.scores, output.sequences, prefix_lengths, logprobs
            )

        if not token_logprobs:
            token_logprobs = None

        return ModelOutput(texts, {"logprobs": token_logprobs})

    def score(self, texts, logprobs: int = 1, num_at_once=5):
        all_logprobs = []
        i = 0
        while i < len(texts):
            curr_texts = texts[i : i + num_at_once]
            inputs = self.tokenizer(curr_texts, padding=True, return_tensors="pt").to(
                self.model.device
            )
            output = self.model(**inputs)
            scores = output.logits[:, :-1, :]
            bs, _, vocab_size = scores.shape
            padding = torch.zeros(bs, 1, vocab_size).to(scores)
            scores = torch.cat([padding, scores], dim=1)
            logprobs = self._get_logprobs(
                scores, inputs.input_ids, [0] * bs, logprobs=logprobs
            )
            i += num_at_once
            all_logprobs.extend(logprobs)
        return all_logprobs

    def _get_logprobs(self, scores, sequences, prefix_lengths, logprobs):
        def id2str(token_id):
            return self.tokenizer.decode(token_id)

        # (|prompts| * n, max_seq_len, vocab_size)
        if isinstance(scores, tuple):
            all_scores = torch.stack(scores, dim=1)
        else:
            all_scores = scores

        all_scores = torch.log_softmax(all_scores, dim=-1)
        # (|prompts| * n, max_seq_len, logprobs)
        topk_scores = torch.topk(all_scores, k=logprobs, dim=-1)
        # only get probabilities up to the end of the sequence
        eos_mask = torch.cumsum(sequences == self.tokenizer.eos_token_id, dim=-1) > 0
        eos_mask = eos_mask.float()
        boundary = torch.ones(eos_mask.shape[:-1]).to(eos_mask).unsqueeze(-1)
        # add 1 to account for EOS token's probability
        # might extend past the end of the tensor but that's ok
        lengths = torch.searchsorted(eos_mask, boundary, right=False).squeeze(-1) + 1

        token_logprobs = []
        for seq_idx in range(len(all_scores)):
            seq_token_logprobs = []
            prefix_length = prefix_lengths[seq_idx]
            length = lengths[seq_idx]
            length = length - prefix_length

            seq_tokens = sequences[seq_idx, prefix_length : prefix_length + length]
            seq_scores = all_scores[seq_idx]

            token_topk_tokens = topk_scores.indices[seq_idx][:length].tolist()
            token_topk_scores = topk_scores.values[seq_idx][:length].tolist()
            for token, token_dist, tokens_, scores_, i in zip(
                seq_tokens,
                seq_scores,
                token_topk_tokens,
                token_topk_scores,
                range(len(seq_tokens)),
            ):
                token_strs = [id2str(token_) for token_ in tokens_]
                top_scores = {}
                for t_, s_ in zip(token_strs, scores_):
                    # Sometimes multiple token ids can map to the same string
                    if t_ not in top_scores:
                        top_scores[t_] = s_

                top_scores[id2str(token)] = token_dist[token].detach().item()
                seq_token_logprobs.append(TokenScore(id2str(token), top_scores))

            token_logprobs.append(seq_token_logprobs)
        return token_logprobs

    def from_pretrained(model_name, **kwargs):
        kwargs["torch_dtype"] = kwargs.get("torch_dtype") or "auto"
        kwargs["cache_dir"] = kwargs.get("cache_dir") or HF_CACHE_DIR

        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        model.to(DEVICE)

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        return HFModel(model, tokenizer)


from torch import FloatTensor, LongTensor


class StopAtStr(StoppingCriteria):
    def __init__(self, stop_str: str, tokenizer: PreTrainedTokenizer, keep_word=True):
        super().__init__()
        self.stop_str = stop_str
        self.prompt_length = -1
        self.tokenizer = tokenizer
        self.dummy_id = tokenizer.eos_token_id or tokenizer.bos_token_id
        self.keep_word = keep_word
        self.done = None

    def __call__(self, input_ids: LongTensor, scores: FloatTensor):
        # initialize some attributes on first call
        if self.done is None:
            self.done = [False] * len(input_ids)
        if self.prompt_length == -1:
            self.prompt_length = input_ids.shape[-1] - 1

        completion_tokens = input_ids[:, self.prompt_length :]
        completions = self.tokenizer.batch_decode(completion_tokens)
        for i, completion in enumerate(completions):
            if self.done[i]:
                input_ids[i, -1] = self.dummy_id
                continue
            if self.stop_str in completion:
                self.done[i] = True
                if not self.keep_word:
                    input_ids[i, -1] = self.dummy_id
        return all(self.done)

        # last_tokens = input_ids[..., -1]
        # input_ids = input_ids.view(-1, input_ids.shape[-1])
        # for i, token in enumerate(last_tokens):
        #     if self.done[i]:
        #         input_ids[i, -1] = self.dummy_id
        #         continue
        #     window_size = min(len(self.word), 4)
        #     end_text = self.tokenizer.decode(input_ids[i, -window_size :])
        #     # print("new word:", token, "\n" + self.tokenizer.decode(token))
        #     if self.word in self.tokenizer.decode([token]) or self.word in end_text:
        #         self.done[i] = True
        #         if not self.keep_word:
        #             input_ids[i, -1] = self.dummy_id
        # return all(self.done)


if __name__ == "__main__":
    model = HFModel.from_pretrained("gpt2", local_files_only=True)
    prompt = ["My favorite show is", "My favorite band is"]
    # (texts, logprobs) = model(prompt, logprobs=3, max_tokens=10, temperature=0.5, top_p=0.95, n=2)
    scores = model.score(prompt)
