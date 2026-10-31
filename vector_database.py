from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
load_dotenv()

# 1. Load PDF
loader = PyPDFLoader("documents/HR_Policy.pdf")
documents = loader.load()

# 2. Split PDF into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)

# 3. Create embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# 4. Store chunks and embeddings in FAISS
vector_db = FAISS.from_documents(
    chunks,
    embeddings
)

# 5. Save the vector database locally
vector_db.save_local("vectorstore/hr_policy")

print("Vector database created successfully!")
# 2. Load the saved FAISS database
vector_db = FAISS.load_local(
    "vectorstore/hr_policy",
    embeddings,
    allow_dangerous_deserialization=True
)



def create_prompt(question, results):

    context = "\n\n".join(
        doc.page_content
        for doc in results
    )

    prompt = f"""
    You are an HR policy assistant.

    Answer the question using ONLY the context below.

    If the answer is not available in the context,
    say you could not find it in the HR policy.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    return prompt

def retrieve_documents(vector_db, question):
# 4. Search for the 2 most relevant chunks
    results = vector_db.similarity_search(
        question,
        k=3
    )

    print("\n--- RETRIEVED DOCUMENTS ---")

    for index, doc in enumerate(results):

        print(f"\nChunk {index + 1}")

        print("Content:", doc.page_content)

        print("Metadata:", doc.metadata)

    return results



def generate_answer(prompt):

    # Connect to locally running Ollama model
    llm = ChatOllama(
        model="llama3.2",
        temperature=0
    )

    # Send prompt to Ollama
    response = llm.invoke(prompt)

    # Return generated answer
    return response.content


question = "How many sick leaves do I get?"

# Stage 1: Retrieve
results = retrieve_documents(
    vector_db,
    question
)

# Stage 2: Construct prompt
prompt = create_prompt(
    question,
    results
)

# Stage 3: Generate answer
answer = generate_answer(prompt)

print("\nFinal Answer:")
print(answer)