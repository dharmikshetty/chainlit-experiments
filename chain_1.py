# from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
# from langchain import HuggingFaceHub
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
import os
import chainlit as cl
# os.environ["HUGGINGFACEHUB_API_TOKEN"]="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"
from langchain_community.embeddings import OllamaEmbeddings

from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
# from langchain_openai import OpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
# from langchain_openai import ChatOpenAI

embeddings = OllamaEmbeddings(model="mistral")




prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="You are a chatbot having a conversation with a human. Answer only in 1 short sentence only"
        ),  # The persistent system prompt
        # MessagesPlaceholder(
        #     variable_name="chat_history"
        # ),  # Where the memory will be stored.
        HumanMessagePromptTemplate.from_template(
            "{input}"
        ),  # Where the human input will injected
    ]
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)









@cl.on_chat_start
async def on_chat_start():
    # model = HuggingFaceHub(repo_id="google/flan-t5-large",model_kwargs={"temperature":0.7,"max_length":64})
    # user = cl.user_session.get("user")
    # chat_profile = cl.user_session.get("chat_profile")
    # await cl.Message(
    #     content=f"starting chat with DHMK using the {chat_profile} chat profile"
    # ).send()
    # if chat_profile=="GPT-3.5"
    model= ChatOllama(model="mistral")
    chat_llm_chain = LLMChain(
    llm=model,
    prompt=prompt,
    verbose=True,
    memory=memory,
)


    msgt = cl.Message(content="This is an IPC CONVERSATIONAL bOT TO Help Guide you")
    await msgt.send()
    await cl.sleep(1)
    await msgt.remove()

    

    runnable = prompt | model | StrOutputParser()
    cl.user_session.set("runnable", runnable)
    


@cl.on_message
async def on_message(message: cl.Message):
    chat_profile = cl.user_session.get("chat_profile")
    await cl.Message(
        content=f"starting chat with DHMK using the {chat_profile} chat profile"
    ).send()
    if chat_profile=="GPT-3.5":
        runnable = cl.user_session.get("runnable")  # type: Runnable

        msg = cl.Message(content="")

        async for chunk in runnable.astream(
            {"input": message.content},
            config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
        ):
            await msg.stream_token(chunk)

        await msg.send()

        # file = await cl.AskFileMessage(
        #     content="Please upload a python file to begin!", accept={"text/plain": [".py"]}
        #     ).send()


@cl.set_chat_profiles
async def chat_profile(current_user: cl.User):
    

    return [
        cl.ChatProfile(
            name="GPT-3.5",
            markdown_description="The underlying LLM model is **GPT-3.5**, a *175B parameter model* trained on 410GB of text data.",
        ),
        cl.ChatProfile(
            name="GPT-4",
            markdown_description="The underlying LLM model is **GPT-4**, a *1.5T parameter model* trained on 3.5TB of text data.",
            icon="https://picsum.photos/250",
        ),
        cl.ChatProfile(
            name="GPT-5",
            markdown_description="The underlying LLM model is **GPT-5**.",
            icon="https://picsum.photos/200",
        ),
    ]