# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This example shows how to use vLLM for running offline inference with
multi-image input on vision language models for text generation,
using the chat template defined by the model.
"""

import os
from argparse import Namespace
from typing import NamedTuple

from huggingface_hub import snapshot_download
from PIL.Image import Image
from transformers import AutoProcessor, AutoTokenizer

from vllm import LLM, EngineArgs, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.multimodal.utils import fetch_image
from vllm.platforms import current_platform
from vllm.utils.argparse_utils import FlexibleArgumentParser

QUESTION = "What is the content of each image?"
IMAGE_URLS = [
    "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/duck.jpg",
    "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/lion.jpg",
    "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/flycatcher.jpeg",
    "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/somefish.jpg",
    "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/starfish.jpg",
    "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/snail.jpg",
    "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/thistle.jpg",
    "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/husky.jpg",
    "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/orangetabbycat.jpg",
    "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/guineapig.jpg",
    "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/rabbit.jpg",
    "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/horsepony.jpg",
]


class ModelRequestData(NamedTuple):
    engine_args: EngineArgs
    prompt: str
    image_data: list[Image]
    stop_token_ids: list[int] | None = None
    chat_template: str | None = None
    lora_requests: list[LoRARequest] | None = None
    sampling_params: SamplingParams | None = None


# NOTE: The default `max_num_seqs` and `max_model_len` may result in OOM on
# lower-end GPUs.
# Unless specified, these settings have been tested to work on a single L4.


def load_aria(question: str, image_urls: list[str]) -> ModelRequestData:
    model_name = "rhymes-ai/Aria"
    engine_args = EngineArgs(
        model=model_name,
        tokenizer_mode="slow",
        trust_remote_code=True,
        dtype="bfloat16",
        limit_mm_per_prompt={"image": len(image_urls)},
    )
    placeholders = "<fim_prefix><|img|><fim_suffix>\n" * len(image_urls)
    prompt = (
        f"<|im_start|>user\n{placeholders}{question}<|im_end|>\n<|im_start|>assistant\n"
    )
    stop_token_ids = [93532, 93653, 944, 93421, 1019, 93653, 93519]

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        stop_token_ids=stop_token_ids,
        image_data=[fetch_image(url) for url in image_urls],
    )




def load_gemma3(question: str, image_urls: list[str]) -> ModelRequestData:
    model_name = "google/gemma-3-4b-it"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=8192,
        max_num_seqs=2,
        limit_mm_per_prompt={"image": len(image_urls)},
    )

    placeholders = [{"type": "image", "image": url} for url in image_urls]
    messages = [
        {
            "role": "user",
            "content": [
                *placeholders,
                {"type": "text", "text": question},
            ],
        }
    ]

    processor = AutoProcessor.from_pretrained(model_name)

    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        image_data=[fetch_image(url) for url in image_urls],
    )


model_example_map = {
    "aria": load_aria,
    "gemma3": load_gemma3,
}


def run_generate(
    model,
    question: str,
    image_urls: list[str],
    seed: int,
    tensor_parallel_size: int | None,
):
    req_data = model_example_map[model](question, image_urls)

    engine_args = req_data.engine_args
    engine_args.seed = seed
    if tensor_parallel_size is not None:
        engine_args.tensor_parallel_size = tensor_parallel_size
    if current_platform.is_rocm():
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    llm = LLM.from_engine_args(engine_args)

    sampling_params = SamplingParams(
        temperature=0.0, max_tokens=256, stop_token_ids=req_data.stop_token_ids
    )

    outputs = llm.generate(
        {
            "prompt": req_data.prompt,
            "multi_modal_data": {"image": req_data.image_data},
        },
        sampling_params=sampling_params,
        lora_request=req_data.lora_requests,
    )

    print("-" * 50)
    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)
        print("-" * 50)


def run_chat(
    model: str,
    question: str,
    image_urls: list[str],
    seed: int,
    tensor_parallel_size: int | None,
):
    req_data = model_example_map[model](question, image_urls)

    # Disable other modalities to save memory
    default_limits = {"image": 0, "video": 0, "audio": 0}
    req_data.engine_args.limit_mm_per_prompt = default_limits | dict(
        req_data.engine_args.limit_mm_per_prompt or {}
    )

    engine_args = req_data.engine_args
    engine_args.seed = seed
    if tensor_parallel_size is not None:
        engine_args.tensor_parallel_size = tensor_parallel_size
    if current_platform.is_rocm():
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    llm = LLM.from_engine_args(engine_args)

    sampling_params = (
        SamplingParams(
            temperature=0.0, max_tokens=256, stop_token_ids=req_data.stop_token_ids
        )
        if req_data.sampling_params is None
        else req_data.sampling_params
    )
    outputs = llm.chat(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": question,
                    },
                    *(
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url},
                        }
                        for image_url in image_urls
                    ),
                ],
            }
        ],
        sampling_params=sampling_params,
        chat_template=req_data.chat_template,
        lora_request=req_data.lora_requests,
    )

    print("-" * 50)
    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)
        print("-" * 50)


def parse_args():
    parser = FlexibleArgumentParser(
        description="Demo on using vLLM for offline inference with "
        "vision language models that support multi-image input for text "
        "generation"
    )
    parser.add_argument(
        "--model-type",
        "-m",
        type=str,
        default="phi3_v",
        choices=model_example_map.keys(),
        help='Huggingface "model_type".',
    )
    parser.add_argument(
        "--method",
        type=str,
        default="generate",
        choices=["generate", "chat"],
        help="The method to run in `vllm.LLM`.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Set the seed when initializing `vllm.LLM`.",
    )
    parser.add_argument(
        "--num-images",
        "-n",
        type=int,
        choices=list(range(1, len(IMAGE_URLS) + 1)),  # the max number of images
        default=2,
        help="Number of images to use for the demo.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        "-tp",
        type=int,
        default=None,
        help="Tensor parallel size to override the model's default setting. ",
    )
    return parser.parse_args()


def main(args: Namespace):
    model = args.model_type
    method = args.method
    seed = args.seed
    tensor_parallel_size = args.tensor_parallel_size

    if tensor_parallel_size is not None and tensor_parallel_size < 1:
        raise ValueError(
            f"tensor_parallel_size must be a positive integer, "
            f"got {tensor_parallel_size}"
        )

    image_urls = IMAGE_URLS[: args.num_images]

    if method == "generate":
        run_generate(model, QUESTION, image_urls, seed, tensor_parallel_size)
    elif method == "chat":
        run_chat(model, QUESTION, image_urls, seed, tensor_parallel_size)
    else:
        raise ValueError(f"Invalid method: {method}")


if __name__ == "__main__":
    args = parse_args()
    main(args)