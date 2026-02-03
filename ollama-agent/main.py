import ollama

response = ollama.chat(model='llama3', messages=[
  {
    'role': 'user',
    'content': 'Write a 1-sentence tagline for a coffee shop.',
  },
])

print(response['message']['content'])