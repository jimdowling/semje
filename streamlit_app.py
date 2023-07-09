import streamlit as st # import the Streamlit library
from langchain.chains import LLMChain, SimpleSequentialChain # import LangChain libraries
from langchain.llms import OpenAI # import OpenAI model
from langchain.prompts import PromptTemplate # import PromptTemplate
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA


# Set the title of the Streamlit app
st.title("✅ Semje Konflikt Utvärdering Verktyg")


option = st.selectbox(
    'Velg din type organisasjonsdokument',
    ('Avslutt intervju', 'Fagfellevurdering', 'Prestasjonsevaluering'))

st.write(option)



persist_directory = "./"


uploaded_file = st.file_uploader("Last opp en fil", "pdf")
if uploaded_file is not None:
    filebytes = uploaded_file.getvalue()
    filename = "doc.pdf"
    with open(filename, 'wb') as f: 
        f.write(filebytes)

    loader = PyMuPDFLoader(filename)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=10)
    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(documents=texts, 
                                    embedding=embeddings,
                                    persist_directory=persist_directory)
    vectordb.persist()

    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(model_name='gpt-4')
# 'gpt-3.5-turbo')

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)


    placeholder = "Følgende er et eksempel på en " + option + ". Vennligst anslå hvor mye konflikt som var en faktor i dette dokumentet." 
    st.write(placeholder)

    query = f"###Prompt {placeholder}"
    try:
        llm_response = qa(query)
        st.write(llm_response["result"])
    except Exception as err:
        st.write('Exception occurred. Please try again', str(err))


