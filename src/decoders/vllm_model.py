import os
import warnings
from typing import Optional

import torch
import vllm
from transformers import SequenceBiasLogitsProcessor

from src.decoders.model_backend import ModelBackend, ModelOutput, TokenScore

HF_CACHE_DIR = os.environ.get("HF_CACHE_DIR")
# Reduce this if experiencing out-of-memory errors
MAX_MODEL_LEN = 3072


def wrap_hf_logits_processor(f):
    return lambda input_ids, scores: f(torch.tensor(input_ids), scores)


class VLLMModel(ModelBackend):
    """
    Wrapper around VLLM model inference
    """

    def __init__(self, vllm_model: vllm.LLM):
        self.vllm_model = vllm_model

    def get_tokenizer(self):
        return self.vllm_model.get_tokenizer()

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
        stop: Optional[list[str]] = None,
        logprobs: int = 0,
        logit_bias: Optional[dict[int, float]] = None,
        seed: Optional[int] = None,
        ### VLLM specific parameters
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        min_p: float = 0.0,
    ) -> ModelOutput:
        logits_processors = []
        if logit_bias is not None:
            logit_bias_processor = SequenceBiasLogitsProcessor(
                {(k,): v for k, v in logit_bias.items()}
            )
            logits_processors.append(wrap_hf_logits_processor(logit_bias_processor))

        if top_k is None:
            top_k = -1

        sampling_config = vllm.SamplingParams(
            n=n,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            stop=stop,
            logprobs=logprobs,
            use_beam_search=use_beam_search,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
            seed=seed,
            logits_processors=logits_processors,
            include_stop_str_in_output=True,
        )

        results = self.vllm_model.generate(prompts, sampling_config, use_tqdm=False)
        texts = []
        for result in results:
            for choice in result.outputs:
                texts.append(choice.text)
        token_logprobs = self._get_logprobs(results, is_prompt=False)
        return ModelOutput(texts, {"logprobs": token_logprobs})

    def _get_logprobs(self, results, is_prompt=False):
        # topk_logprobs = []
        token_logprobs = []

        tokenizer = self.get_tokenizer()

        def id2str(token_id):
            return tokenizer.decode(token_id)

        for result in results:
            if is_prompt:
                seq_token_logprobs = []
                for token_id, token_top_logprobs in zip(
                    result.prompt_token_ids, result.prompt_logprobs
                ):
                    if token_top_logprobs is None:
                        token_top_logprobs = {token_id: 0.0}
                    top_scores = {}
                    for id_, s_ in token_top_logprobs.items():
                        t_ = id2str(id_)
                        if t_ not in top_scores:
                            top_scores[t_] = s_
                    seq_token_logprobs.append(TokenScore(id2str(token_id), top_scores))
                token_logprobs.append(seq_token_logprobs)
            else:
                for choice in result.outputs:
                    key = "prompt_logprobs" if is_prompt else "logprobs"
                    if getattr(choice, key):
                        # curr_topk_logprobs = [sorted(token_logprobs.items(), key=lambda t: -t[1])
                        #                       for token_logprobs in choice.logprobs]
                        seq_token_logprobs = [
                            TokenScore(
                                id2str(token_id),
                                {id2str(k): v for k, v in token_logprobs.items()},
                            )
                            for token_id, token_logprobs in zip(
                                choice.token_ids, choice.logprobs
                            )
                            # token_logprobs[token_id]
                            # for token_id, token_logprobs in zip(choice.token_ids, choice.logprobs)
                        ]
                        token_logprobs.append(seq_token_logprobs)
                        # token_logprobs.append(curr_token_logprobs)
                        # topk_logprobs.append(curr_topk_logprobs)
        # if not topk_logprobs:
        #     topk_logprobs = None
        if not token_logprobs:
            token_logprobs = None
        return token_logprobs

    def score(self, texts, logprobs: int = 1):
        sampling_params = vllm.SamplingParams(max_tokens=1, prompt_logprobs=logprobs)
        results = self.vllm_model.generate(texts, sampling_params, use_tqdm=False)
        return self._get_logprobs(results, is_prompt=True)

    @staticmethod
    def from_pretrained(model_name: str, **kwargs):
        kwargs["download_dir"] = kwargs.get("download_dir") or HF_CACHE_DIR
        kwargs["max_model_len"] = kwargs.get("max_model_len") or MAX_MODEL_LEN
        vllm_model = vllm.LLM(model_name, **kwargs)
        return VLLMModel(vllm_model)


if __name__ == "__main__":
    model = VLLMModel.from_pretrained(
        "microsoft/phi-2", max_model_len=512, dtype=torch.bfloat16
    )
    prompt = ["My favorite show is", "My favorite band is"]
    tokenizer = model.get_tokenizer()
    prompt = [f"{tokenizer.bos_token}{p}" for p in prompt]

    scores = model.score(prompt, logprobs=10)

    texts, logprobs = model(
        prompt, logprobs=3, max_tokens=10, temperature=0.5, top_p=0.95, n=2
    )
