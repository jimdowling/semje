import streamlit as st # import the Streamlit library
from langchain.chains import LLMChain, SimpleSequentialChain # import LangChain libraries
from langchain.llms import OpenAI # import OpenAI model
from langchain.prompts import PromptTemplate # import PromptTemplate
from langchain.document_loaders import PyMuPDFLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

import os


# Set the title of the Streamlit app
st.title("✅ Semje Konflikt Utvärdering Verktyg")

OPEN_API_KEY=os.environ['OPENAI_API_KEY']

# llm = OpenAI(temperature=0.7, openai_api_key=OPEN_API_KEY)

llm = ChatOpenAI(model_name='gpt-4', openai_api_key=OPEN_API_KEY)


option = st.selectbox(
    'Select your type of organization document',
    ('Exit Interview', 'Peer Review', 'Performance Review'))

st.write('You selected:', option)


user_question = st.text_area(
# st.text_input(
    "Enter Your document text: ",
    placeholder = "Paste in the contents of the document here",
)

placeholder = "The following is an example of a " + option + ". Please estimate how much conflict was a factor in this document. " + user_question


# from langchain.document_loaders import PyMuPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import Chroma
# from langchain.embeddings import OpenAIEmbeddings
# loader = PyMuPDFLoader("./docs/example.pdf")
# documents = loader.load()
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=10)
# texts = text_splitter.split_documents(documents)
# persist_directory = "./storage"
# embeddings = OpenAIEmbeddings()
# vectordb = Chroma.from_documents(documents=texts, 
#                                  embedding=embeddings,
#                                  persist_directory=persist_directory)
# vectordb.persist()


persist_directory = "./"


# loader = PyPDFLoader("example_data/layout-parser-paper.pdf")
# pages = loader.load_and_split()

uploaded_file = st.file_uploader("Choose a file", "pdf")
if uploaded_file is not None:
    # file_details = {"FileName":"doc.pdf","FileType":"pdf"}
    # st.write(file_details)
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

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)


    # user_input = st.text_input(
    #     "Enter a question about your documents",
    #     placeholder = "here",
    # )

    placeholder = "The following is an example of a " + option + ". Please estimate how much conflict was a factor in this document. " + user_question
    st.write(placeholder)

    query = f"###Prompt {placeholder}"
    try:
        llm_response = qa(query)
        st.write(llm_response["result"])
    except Exception as err:
        st.write('Exception occurred. Please try again', str(err))



# if st.button("Evaluate", type="primary"):

#     query = f"###Prompt {user_input}"
#     try:
#         llm_response = qa(query)
#         print(llm_response["result"])
#     except Exception as err:
#         print('Exception occurred. Please try again', str(err))

    # # Chain 1: Generating a rephrased version of the user's question
    # template = """{question}\n\n"""
    # prompt_template = PromptTemplate(input_variables=["question"], template=template)
    # question_chain = LLMChain(llm=llm, prompt=prompt_template)

    # # Chain 2: Generating assumptions made in the statement
    # template = """Here is a statement:
    #     {statement}
    #     Make a bullet point list of the assumptions you made when producing the above statement.\n\n"""
    # prompt_template = PromptTemplate(input_variables=["statement"], template=template)
    # assumptions_chain = LLMChain(llm=llm, prompt=prompt_template)
    # assumptions_chain_seq = SimpleSequentialChain(
    #     chains=[question_chain, assumptions_chain], verbose=True
    # )

    # # Chain 3: Fact checking the assumptions
    # template = """Here is a bullet point list of assertions:
    # {assertions}
    # For each assertion, determine whether it is true or false. If it is false, explain why.\n\n"""
    # prompt_template = PromptTemplate(input_variables=["assertions"], template=template)
    # fact_checker_chain = LLMChain(llm=llm, prompt=prompt_template)
    # fact_checker_chain_seq = SimpleSequentialChain(
    #     chains=[question_chain, assumptions_chain, fact_checker_chain], verbose=True
    # )

    # # Final Chain: Generating the final answer to the user's question based on the facts and assumptions
    # template = """In light of the above facts, how would you answer the question '{}'""".format(
    #     user_question
    # )
    # template = """{facts}\n""" + template
    # prompt_template = PromptTemplate(input_variables=["facts"], template=template)
    # answer_chain = LLMChain(llm=llm, prompt=prompt_template)
    # overall_chain = SimpleSequentialChain(
    #     chains=[question_chain, assumptions_chain, fact_checker_chain, answer_chain],
    #     verbose=True,
    # )

    # # Running all the chains on the user's question and displaying the final answer
    # st.success(overall_chain.run(user_question))
