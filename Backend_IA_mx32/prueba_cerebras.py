from cerebras.cloud.sdk import Cerebras

client = Cerebras(
    # This is the default and can be omitted
    api_key="csk-5cn5fvvvdn26t3jw44pf2d5e3cdykvjmcvr4yc3jk2et9njj"
)

stream = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "hola bb dime que me quieres mucho mi amor"
        }
    ],
    model="gpt-oss-120b",
    stream=True,
    max_completion_tokens=65536,
    temperature=1,
    top_p=1,
    reasoning_effort="high"
)

for chunk in stream:
  print(chunk.choices[0].delta.content or "", end="")
  
