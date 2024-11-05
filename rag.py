import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_cohere.llms import Cohere
from langchain_cohere import CohereEmbeddings

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.vectorstores import Qdrant

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

from qdrant_client import models, QdrantClient
from database.constants import QDRANT_URL, QDRANT_API_KEY
from transformers import AutoTokenizer, AutoModelForCausalLM

import getpass
import os



from langchain_core.messages import HumanMessage
from database.constants import COHERE_API_KEY

os.environ["COHERE_API_KEY"] = COHERE_API_KEY
model = Cohere(max_tokens=256, temperature=0.75)
message = "Knock knock"
print(model.invoke(message))
print("hello")

# from transformers import pipeline

# pl = pipeline("text-generation", model="medalpaca/medalpaca-7b", tokenizer="medalpaca/medalpaca-7b")
# question = "What are the symptoms of diabetes?"
# context = "Diabetes is a metabolic disease that causes high blood sugar. The symptoms include increased thirst, frequent urination, and unexplained weight loss."
# answer = pl(f"Context: {context}\n\nQuestion: {question}\n\nAnswer: ")
# print(answer)
# from transformers import pipeline

# pl = pipeline("text-generation", model="gpt2", tokenizer="gpt2")
# question = "What are the symptoms of diabetes?"
# context = "Diabetes is a metabolic disease that causes high blood sugar. The symptoms include increased thirst, frequent urination, and unexplained weight loss."
# answer = pl(f"Context: {context}\n\nQuestion: {question}\n\nAnswer: ")
# print(answer)

from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("book.pdf")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(documents)
print(splits)

from sentence_transformers import SentenceTransformer
from langchain_qdrant import QdrantVectorStore

embeddings_model = "sentence-transformers/all-MiniLM-L6-v2"
# embedding_model = CohereEmbeddings(model="embed-english-v3.0", cohere_api_key=os.getenv("COHERE_API_KEY"))
# embedding = HuggingFaceEmbeddings(model_name=embeddings_model)
embedding = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=60
    )
print(client.get_collections())
#client.create_collection(
 #        collection_name="XRAYS",
 #        vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),
 #    )
vector_store = QdrantVectorStore.from_documents(documents, embedding, url=QDRANT_URL, api_key=QDRANT_API_KEY, collection_name="Xrays")

records_to_upload = []
print(client.get_collections())
for idx, chunk in enumerate(documents):
    content = chunk.page_content
    # Change how you call to get the embedding
    # vector = embedding.embed_documents([content])[0]  # Check if this is the correct call
    embeddings= embedding.encode(content).tolist()
    record = models.PointStruct(
        id=idx,
        vector=embeddings,
        payload={"page_content": content}
    )
    records_to_upload.append(record)
print(records_to_upload[:5])
client.upload_points(
    collection_name="XRAYS2",
    points=records_to_upload
)

retriever = vector_store.as_retriever()
prompt = hub.pull("rlm/rag-prompt")
prompt

def format_docs(docs):
  return "\n\n".join(doc.page_content for doc in docs)
rag_chain = (
{"context": retriever, "question": RunnablePassthrough()}
| prompt
| model
| StrOutputParser()
)

rag_chain.invoke("explain pneumothorax")

rag_chain.invoke("Pneumothorax")
