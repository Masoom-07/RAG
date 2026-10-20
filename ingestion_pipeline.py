
import os
import time 

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

def load_documents(docs_path="docs"):
    """Load all text files from the docs directory"""
    print(f"Loading documents from {docs_path}...")
    
    # Check if docs directory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist. Please create it and add your company files.")
    
    # Load all .txt files from the docs directory
    loader = DirectoryLoader(
    path=docs_path,
    glob="*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"},
    silent_errors=False,
)
    
    documents = loader.load()
    
    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in {docs_path}. Please add your company documents.")
    
   
    # for i, doc in enumerate(documents[:2]):  # Show first 2 documents
    #     print(f"\nDocument {i+1}:")
    #     print(f"  Source: {doc.metadata['source']}")
    #     print(f"  Content length: {len(doc.page_content)} characters")
    #     print(f"  Content preview: {doc.page_content[:100]}...")
    #     print(f"  metadata: {doc.metadata}")

    return documents


def split_documents(documents, chunk_size=1000, chunk_overlap=150):
    """Split documents into smaller, overlapping chunks."""
    print("Splitting documents into chunks...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    return chunks



def create_vector_store(chunks, persist_directory="db/chroma_gemini_v2"):
    """Create ChromaDB and add documents in controlled batches."""
    print("Creating embeddings and storing in ChromaDB...")

    embedding_model = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001"
    )

    vectorstore = Chroma(
        collection_name="rag_documents",
        embedding_function=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )

    batch_size = 50

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]

        print(
            f"Processing chunks {start + 1} "
            f"to {start + len(batch)} of {len(chunks)}..."
        )

        vectorstore.add_documents(batch)

        if start + batch_size < len(chunks):
            print("Waiting 60 seconds before the next batch...")
            time.sleep(60)

    print("Finished creating vector store.")
    print(f"Vector store saved to {persist_directory}")

    return vectorstore


def main():
    print("main function ")
 # for loading the documents for chunking

    documents =load_documents(docs_path="docs")

# chunking the files
    chunks=split_documents(documents)
     
#adding the embedding model and addint the data in vectorDB

    vectorstore=create_vector_store(chunks);



if __name__=="__main__":
    main()   