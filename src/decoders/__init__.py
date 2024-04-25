from src.decoders.model_backend import ModelBackend


def build_decoder(
    decoder_name: str, decoder_backend: str, **decoder_kwargs
) -> ModelBackend:
    """
    Constructs LM decoder for text completion

    Args:
        decoder_name: Name of the decoder. Can be a HuggingFace model name
            or the name of an OpenAI model.
        decoder_backend: Library/service for model inference. Currently only
            supports HF, vLLM and OpenAI.
        **decoder_kwargs: Dict of backend-specific args for decoder
    """
    match decoder_backend:
        case "vllm":
            from src.decoders.vllm_model import VLLMModel as ModelCls
        case "openai":
            from src.decoders.openai_model import OpenAIModel as ModelCls
        case "hf":
            from src.decoders.hf_model import HFModel as ModelCls
        case _:
            raise Exception(f"Unknown model backend: {decoder_backend}")

    return ModelCls.from_pretrained(decoder_name, **decoder_kwargs)
