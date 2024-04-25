import abc
import copy
from collections import namedtuple
from dataclasses import asdict, dataclass
from typing import Any, Optional

ModelOutput = namedtuple("ModelOutput", ["texts", "scores"])


@dataclass
class ModelOutput:
    """
    texts: list[str]
        List of text completions

    scores: dict[str, list[Any]]
        Dict mapping score name (i.e. log probabilities) to list of scores for each completion.
        This score list will usually be a list of floats (i.e. one float score per choice).
        However, for 'logprobs', this list will be a list of top probabilities per token,
        stored in TokenScore objects, i.e. list[list[TokenScore]].
    """

    texts: list[str] = None
    scores: dict[str, list[Any]] = None

    def split_batch(self, batch_size: int) -> list[ModelOutput]:
        """
        batch_size: int
            Number of examples in batch
        """
        # n is number of generated sequences per example in the batch
        n = len(self.texts) // batch_size
        batches = []
        for i in range(batch_size):
            curr_texts = self.texts[i * n : (i + 1) * n]
            curr_scores = dict()
            for k, scores in self.scores.items():
                curr_scores[k] = scores[i * n : (i + 1) * n]
            batches.append(ModelOutput(curr_texts, curr_scores))
        return batches

    @staticmethod
    def merge_batch(batch_outputs: list[ModelOutput]) -> ModelOutput:
        texts = []
        scores = {k: [] for k in batch_outputs[0].scores}
        for output in batch_outputs:
            texts.extend(output.texts)
            for k, scores_ in output.scores.items():
                scores[k].extend(scores_)
        return ModelOutput(texts, scores)

    def __iter__(self):
        yield self.texts
        yield self.scores

    def to_dict(self):
        texts = copy.deepcopy(self.texts)
        scores = copy.deepcopy(self.scores)
        if "logprobs" in self.scores:
            logprobs = []
            for seq_logprobs in self.scores["logprobs"]:
                logprobs.append([tuple(t) for t in seq_logprobs])
            scores["logprobs"] = logprobs
        return {"texts": texts, "scores": scores}

    @staticmethod
    def from_dict(d):
        texts = d["texts"]
        scores = d["scores"]
        if "logprobs" in scores:
            logprobs = scores["logprobs"]
            for i in range(len(logprobs)):
                logprobs[i] = [TokenScore(*v) for v in logprobs[i]]
        return ModelOutput(texts, scores)


class ModelBackend(abc.ABC):
    """
    Abstract base class for LM inference wrappers
    """

    @abc.abstractmethod
    def generate(
        # a list of arguments supported by *most* model services
        # (but not all, i.e. openai doesn't support beam search)
        self,
        prompts: list[str],
        *,  # all of the following arguments are keyword-only
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
    ) -> ModelOutput:
        raise NotImplementedError

    def __call__(self, prompts: list[str], **kwargs) -> ModelOutput:
        generation_config = getattr(self, "generation_config", {})
        kwargs = {
            **generation_config,
            # allow generation_config to be overriden by arguments to method
            **kwargs,
        }

        if isinstance(prompts, str):
            prompts = [prompts]

        cached_outputs = getattr(self, "cached_outputs", None)
        if cached_outputs is not None:
            cached_outputs = [
                ModelOutput.from_dict(next(self.cached_outputs))
                for _ in range(len(prompts))
            ]
            return ModelOutput.merge_batch(cached_outputs)

        return self.generate(prompts, **kwargs)

    @abc.abstractmethod
    def get_tokenizer(self):
        raise NotImplementedError

    def set_generation_config(self, generation_config: dict[str, Any]):
        self.generation_config = generation_config

    def set_cached_outputs(self, cached_outputs: list[dict]):
        """
        Gives model a cache of past outputs; assume the inputs fed to the model
        are in the same order as the cached outputs
        """
        self.cached_outputs = iter(cached_outputs)

    @staticmethod
    @abc.abstractmethod
    def from_pretrained(model_name, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _make_messages(prompt: str):
        return [
            {"role": "user", "content": prompt},
        ]


@dataclass
class TokenScore:
    def __init__(self, token: str, top_scores: dict[str, float]):
        """
        Dataclass to hold per-token model scores (i.e. logprobs).
        Added syntactic sugar to support:

        +/- operations:
            >>> a = TokenScore("a", {"a": 0.2, "b": 0.1})
            >>> b = TokenScore("b", {"b": 0.1})
            >>> a + b
            0.30000000000000004
        string conversion:
            >>> tokens = [a, b]
            >>> ''.join(str(t) for t in tokens)
            'ab'
        2-tuple unpacking:
            >>> x, y = a
        """
        self.top_scores = top_scores
        self.token = token
        self.token_score = top_scores[token]

    def __add__(self, other):
        return self.token_score + float(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.token_score - float(other)

    def __rsub__(self, other):
        return -self.__sub__(other)

    def __float__(self):
        return float(self.token_score)

    def __repr__(self):
        return self.token

    def __iter__(self):
        yield self.token
        yield self.top_scores


if __name__ == "__main__":
    a = TokenScore("a", {"a": 0.2})
    b = TokenScore("b", {"b": 0.1})
    print(a + b, 5 + a, a + 5)
    print(a - b, 5 - a, a - 5)

    tokens = [a, b]
    print("".join(str(t) for t in tokens))
