import os
from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

# Single-image input inference

# Public image URL for testing remote image processing
image_url = "https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"

# Create chat completion with remote image
chat_response = client.chat.completions.create(
    model="microsoft/Phi-3.5-vision-instruct",
    messages=[
        {
            "role": "user",
            "content": [
                # NOTE: The prompt formatting with the image token `<image>` is not needed
                # since the prompt will be processed automatically by the API server.
                {
                    "type": "text",
                    "text": "What’s in this image?",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                    "uuid": image_url,  # Optional
                },
            ],
        }
    ],
)
print("Chat completion output:", chat_response.choices[0].message.content)
