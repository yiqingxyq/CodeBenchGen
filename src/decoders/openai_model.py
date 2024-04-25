import os
import time
import warnings
from typing import Optional

import openai
import tiktoken
from openai import OpenAI
from transformers import AutoTokenizer

from src.decoders.model_backend import ModelBackend, ModelOutput, TokenScore

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

MAX_ATTEMPTS = 2
BACKOFF_RATIO = 1.5


class OpenAIModel(ModelBackend):
    """
    Wrapper around OpenAI (and OpenAI API compatible) model inference
    """

    def __init__(self, model_name, client):
        self.client = client
        self.model_name = model_name
        try:
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        except:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

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
        ### OpenAI specific parameters
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
    ) -> ModelOutput:  # TODO: raises openai.OpenAIError
        texts = []
        token_logprobs = []

        if use_beam_search:
            warnings.warn("OpenAI models do not support beam search!")
        if top_k is not None:
            warnings.warn("OpenAI models do not support top-k!")

        generation_config = {
            "n": n,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stop": stop,
            "logprobs": logprobs > 0,
            **({"top_logprobs": logprobs} if logprobs else {}),
            "max_tokens": max_tokens,
            "logit_bias": logit_bias,
            "seed": seed,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
        }
        for prompt in prompts:
            if not isinstance(prompt[0], dict):
                prompt = self._make_messages(prompt)

            curr_texts, curr_token_logprobs = self._get_chat_response(
                prompt, **generation_config
            )

            texts.extend(curr_texts)
            if curr_token_logprobs:
                token_logprobs.extend(curr_token_logprobs)
            # if curr_topk_logprobs:
            #     topk_logprobs.extend(curr_topk_logprobs)

        if not token_logprobs:
            token_logprobs = None

        return ModelOutput(texts, {"logprobs": token_logprobs})

    def get_tokenizer(self):
        return self.tokenizer

    @staticmethod
    def from_pretrained(model_name, **kwargs):
        client = OpenAI(api_key=OPENAI_API_KEY, **kwargs)
        return OpenAIModel(model_name, client)

    def _get_chat_response(self, messages: list[dict[str, str]], **generation_config):
        response = None
        wait_time = 5
        for attempt in range(MAX_ATTEMPTS):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name, messages=messages, **generation_config
                )
                break
            except openai.OpenAIError as e:
                if attempt == MAX_ATTEMPTS - 1:
                    raise e
                warnings.warn(f"Caught OpenAI error: {e}")
                time.sleep(wait_time)
                wait_time *= BACKOFF_RATIO

        texts = []
        token_logprobs = []
        for choice in response.choices:
            texts.append(choice.message.content)
            if getattr(choice, "logprobs", None):
                seq_token_logprobs = []
                for token_score in choice.logprobs.content:
                    top_scores = {
                        score.token: score.logprob for score in token_score.top_logprobs
                    }
                    top_scores[token_score.token] = token_score.logprob
                    seq_token_logprobs.append(TokenScore(token_score.token, top_scores))
                    # token_topk_logprobs.append([(score.token, score.logprob) for score in token_score.top_logprobs])

                token_logprobs.append(seq_token_logprobs)
                # batch_token_logprobs.append(token_logprobs)
                # batch_topk_logprobs.append(token_topk_logprobs)

        return texts, token_logprobs


if __name__ == "__main__":
    model = OpenAIModel.from_pretrained("gpt-3.5-turbo")
    prompt = ["Large language models are", "Few shot learning is"]
    output, logprobs = model(
        prompt, logprobs=3, max_tokens=10, temperature=0.5, top_p=0.95, n=2
    )
