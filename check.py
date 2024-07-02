from langchain_community.llms import Ollama

llm=Ollama(model="mistral")

response=llm.invoke("Explain me Word war 2")

print(response)