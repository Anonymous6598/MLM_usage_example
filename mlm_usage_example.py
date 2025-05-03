import intel_npu_acceleration_library, torch, warnings, transformers, typing

warnings.filterwarnings(f"ignore")

def initialize_model() -> transformers.Pipeline:
    global stream
    model_id: str = f"microsoft/Phi-3-medium-128k-instruct"
    model: typing.Any = transformers.AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, use_cache=True).eval()
    tokenizer: transformers.PreTrainedTokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    model: typing.Any = intel_npu_acceleration_library.compile(model, dtype=torch.float16)
    pipe: transformers.Pipeline = transformers.pipeline(f"text-generation", model=model, tokenizer=tokenizer)
    stream = transformers.TextStreamer(tokenizer, skip_special_tokens=True)
    return pipe

def main(pipe: transformers.Pipeline, prompt: str) -> str:
    generation_args: dict[str: int, str: bool, str: float, str: bool, str: transformers.TextStreamer] = {f"max_new_tokens": 1024, f"return_full_text": False, f"temperature": 0.3, f"do_sample": False, f"streamer": stream}
    query = f"<|system|>You are a helpful AI assistant.<|end|><|user|>{prompt}<|end|><|assistant|>"
    output = pipe(query, **generation_args)
    return output[0][f"generated_text"]

if __name__ == f"__main__":
    pipe = initialize_model()
    while True:
        prompt: str = input(f"Enter your prompt: ")
        print(main(pipe, prompt))
