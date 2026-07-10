#!/usr/bin/env python

import sys
from pathlib import Path

import ollama


def describe_image(image_path: str) -> str:
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    response = ollama.chat(
        model="llava",
        messages=[
            {
                "role": "user",
                "content": "Describe this image in detail.",
                "images": [str(path)],
            }
        ],
    )

    return response["message"]["content"]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python describe_image.py /path/to/image.jpg")
        sys.exit(1)

    description = describe_image(sys.argv[1])
    print(description)
