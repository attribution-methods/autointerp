curl -X POST "https://spar-agent-resource.services.ai.azure.com/anthropic/v1/messages" \
  -H "Content-Type: application/json" \
  -H "x-api-key: YOUR_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "max_tokens": 1000,
    "temperature": 0.7,
    "system": "You are a helpful assistant.",
    "messages": [
      { "role": "user", "content": "What are 3 things to visit in Seattle?" }
    ]
    "model": "claude-haiku-4-5"
  }'