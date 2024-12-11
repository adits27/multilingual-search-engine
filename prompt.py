from langchain_huggingface.llms import HuggingFacePipeline
from transformers import pipeline
import os
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


PROMPT = '''
You are a Wikipedia chatbot that has answers to all questions. Leverage your knowledge and answer the questions asked in a very concise manner. Use the context shared below and answer the question that follows it. -  \n: 
Context: {context} \n
Question: {question}
'''

FAISS_INDEX_PATH = os.path.dirname(os.path.realpath(__file__)) + "/faiss_index"

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def prompt(question):

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True )

    retriever = db.as_retriever(search_kwargs={"k": 4}, search_type="mmr")

    qaprompt = PromptTemplate(input_variables=["context", "question"], template=PROMPT)

    model_id = "meta-llama/Llama-3.2-3B-Instruct"

    model_kwargs = {
            "offload_folder": "offload",
            "low_cpu_mem_usage" : "True",
            "temperature": 0.2,
            "do_sample": True,
            "max_length": 800,
        }

    pipe = pipeline(
            task = "text-generation",
            model = model_id,
            max_new_tokens = 800,
            top_p = 1,
            use_fast = True,
            top_k = 10,
            model_kwargs=model_kwargs,
            return_full_text = False,
            repetition_penalty = 1.1,
        )
    
    pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id[0]

    llm = HuggingFacePipeline(
        pipeline=pipe,
        batch_size=1,
    )

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | qaprompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain.invoke(question)