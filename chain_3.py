from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
# from langchain import HuggingFaceHub
from langchain_community.llms import Ollama
import os
import chainlit as cl
# os.environ["HUGGINGFACEHUB_API_TOKEN"]="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"

@cl.on_chat_start
async def on_chat_start():
    # model = HuggingFaceHub(repo_id="google/flan-t5-large",model_kwargs={"temperature":0.7,"max_length":64})
    model= Ollama(model="mistral")
    prompt = ChatPromptTemplate.from_messages(
         [
        (
            "system",
            '''You are a virtual assistant chatbot providing a conversational interface for collecting data from user inputs for specific fields which are
            listed down below. The objective is to collect this information as input data and store it in a suitable data structure for future use.
            1) Product
            2) Nature of Deployment: There are 4 types of deployments: VM + IDC BW, VM + IDC + Hybrid Connectivity, VM + Hybrid Connectivity for BYOC, VM + Hybrid Connectivity with VPN. Provide the user with 4 options.
            3) Number of Virtual Machines (VMs)
            4) OS Disk Size
            5) kDUMP Size (in GB)
            6) Additional Disk Size
            7) SFDC ID
            8) Quote ID
            If the user does not provide all the information then you should ask the user
            to give all remaining inputs unless all inputs have been provided.
            Only after all 8 field inputs have been provided, store these inputs in a suitable data structure format like a Python dictionary.
            Answer all questions to the best of your ability.''',
        ),
        ("user", "{input}")
    ]
    )
    runnable = prompt | model | StrOutputParser()
    cl.user_session.set("runnable", runnable)


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")  # type: Runnable

    msg = cl.Message(content="")

    async for chunk in runnable.astream(
        {"input": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()
