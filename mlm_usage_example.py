import intel_npu_acceleration_library, torch, warnings, transformers, typing

warnings.filterwarnings(f"ignore")

def initialize_model() -> transformers.Pipeline:
    model_id: str = f"microsoft/Phi-3-medium-128k-instruct"
    model: typing.Any = transformers.AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, use_cache=True, trust_remote_code=True).eval()
    tokenizer: transformers.PreTrainedTokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    model: typing.Any = intel_npu_acceleration_library.compile(model, dtype=torch.float32)
    pipe: transformers.Pipeline = transformers.pipeline(f"text-generation", model=model, tokenizer=tokenizer)
    return pipe

def main(pipe: transformers.Pipeline, prompt: str) -> str:
    generation_args: dict[str: int, str: bool, str: float, str: bool] = {f"max_new_tokens": 128_000, f"return_full_text": False, f"temperature": 0.3, f"do_sample": False}
    query = f"<|system|>You are a helpful AI assistant.<|end|><|user|>{prompt}<|end|><|assistant|>"
    output = pipe(query, **generation_args)
    return output[0][f"generated_text"]

if __name__ == f"__main__":
    pipe = initialize_model()
    while True:
        prompt: str = input(f"Enter your prompt: ")
        print(main(pipe, prompt))
