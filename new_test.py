import argparse
import openvino_genai

def simple_example(prompt: str) -> str:
    pipe: openvino_genai.LLMPipeline = openvino_genai.LLMPipeline(f"microsoft/Phi-3-medium-128k-instruct", device=f"NPU")

    while True:
        prompt: str = input(f"Enter your prompt: ")
        return pipe.generate(prompt, max_new_tokens=1024, temperature=0.3)

def complex_example(prompt: str) -> str:
    parser: openvino_genai.GenerationConfig = argparse.ArgumentParser()
    parser.add_argument(f"path", help=f"Path to the model directory")
    parser.add_argument(f"device", nargs=f"?", default=f"CPU", help=f"Device to run the model on (default: CPU)")
    args: argparse.Namespace = parser.parse_args()

    device: str = args.device
    path: str = args.path
    pipe: openvino_genai.LLMPipeline = openvino_genai.LLMPipeline(path, device)

    config: openvino_genai.GenerationConfig = openvino_genai.GenerationConfig()
    config.max_new_tokens: int = 1024
    config.temperature: float = 0.3
    config.top_p: float = 0.95
    config.top_k: int = 50
    config.repetition_penalty: float = 1.3
    config.num_beams: int = 1
    config.num_return_sequences: int = 1
    config.do_sample: bool = False

    result: str = pipe.generate(prompt, config)
    return result

if f"__main__" == __name__:
    while True:
        print(complex_example(prompt := input(f"Enter your prompt: ")))
