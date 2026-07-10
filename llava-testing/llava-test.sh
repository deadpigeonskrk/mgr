#!/bin/bash

curl http://localhost:11434/api/chat -d '{
  "model": "llava:34b",
  "prompt": "Describe the image",
  "images": ["./example.jpg"]
}'
