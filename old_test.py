import intel_npu_acceleration_library, torch, transformers, typing

def initialize_model() -> transformers.Pipeline:
    model_id: str = f"microsoft/Phi-3-medium-128k-instruct"
    model: typing.Any = transformers.AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, use_cache=True).eval()
    tokenizer: transformers.PreTrainedTokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    model: typing.Any = intel_npu_acceleration_library.compile(model, dtype=torch.float16)
    pipe: transformers.Pipeline = transformers.pipeline(f"text-generation", model=model, tokenizer=tokenizer)
    return pipe

def main(pipe: transformers.Pipeline, prompt: str) -> str:
    generation_args: dict[str: int, str: bool, str: float, str: bool] = {f"max_new_tokens": 1024, f"return_full_text": False, f"temperature": 0.3, f"do_sample": False}
    query: str = f"<|system|>You are a helpful AI assistant.<|end|><|user|>{prompt}<|end|><|assistant|>"
    output: str = pipe(query, **generation_args)
    return output[0][f"generated_text"]

if __name__ == f"__main__":
    pipe = initialize_model()
    while True:
        prompt: str = input(f"Enter your prompt: ")
        print(main(pipe, prompt))
