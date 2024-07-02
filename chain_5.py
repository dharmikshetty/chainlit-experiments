from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
import os
import io
import chainlit as cl
import shutil
from pypdf import PdfReader


os.environ["OPENAI_API_KEY"] = "{Your_API_Key}"

@cl.on_chat_start
def init():
    file = None
      while files is None:
        files = await cl.AskFileMessage(
            content="Please upload a pdf file to begin!",
            accept=["pdf"]
        ).send()

    # Check if files is a list or a single file object
    if isinstance(files, list):
        # Multiple files uploaded, use the first one
        file = files[0]
    else:
        # Single file uploaded
        file = files

    # Save the file to disk
    file_path = Path("./doc/doc.pdf")
    file_path.parent.mkdir(exist_ok=True)
    with open(file_path, 'wb') as f:
        f.write(file.content)









    # Wait for the user to upload a file
    while file == None:
        file = cl.AskFileMessage(
            content="Please upload a pdf file to begin!", accept=["pdf"]
        ).send()

    with open("./doc/doc.pdf", 'wb') as f: 
        f.write(file.content)

    text = ''
    reader = PdfReader("./doc/doc.pdf")
    for i in range(len(reader.pages)):
        text += reader.pages[i].extract_text()

    shutil.rmtree('./.chroma', ignore_errors=True)

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    docs = text_splitter.split_text(text)

    # Create a metadata for each chunk
    metadatas = [{"source": f"{i}-pl"} for i in range(len(docs))]

    # Create a Chroma vector store
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_texts(docs, embeddings, metadatas=metadatas)

    # Create a chain that uses the Chroma vector store
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        ChatOpenAI(temperature=0),
        chain_type="stuff",
        retriever=db.as_retriever(),
    )

    cl.user_session.set("texts", docs)

    cl.Message(content=f"`{file.name}` is ready for queries!").send()

    return chain



# @cl.langchain_postprocess
@cl.on_message
def process_response(res):
    answer = res["answer"]
    sources = res["sources"].strip().split(',')
    source_elements = []
    
    texts = cl.user_session.get("texts")

    for s in sources:
    # Extract the part of the substring before "-pl"
        index = int(s[:s.find('-pl')])
        source_elements.append(cl.Text(text=texts[index], name=s))
        
    response = f"{answer} Sources: {res['sources']}"
    cl.Message(content=response, elements=source_elements).send()