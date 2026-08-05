# This is just an example API call from OpenAI's Python SDK

# pip install openai

from openai import OpenAI

client = OpenAI(
  api_key="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"
)

response = client.responses.create(
  model="gpt-5.4-mini",
  input="write a haiku about ai",
  store=True,
)

print(response.output_text);
