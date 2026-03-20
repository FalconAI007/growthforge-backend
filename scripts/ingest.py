from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_NAME = os.getenv("PINECONE_INDEX", "growthforge")


def get_or_create_index():
    existing = [i.name for i in pc.list_indexes()]
    if INDEX_NAME not in existing:
        print(f"Creating Pinecone index '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print(f"✅ Index '{INDEX_NAME}' created")
    else:
        print(f"✅ Using existing index '{INDEX_NAME}'")
    return pc.Index(INDEX_NAME)


def ingest(file_path, doc_type, index):
    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()

    for doc in docs:
        doc.metadata["type"] = doc_type
        doc.metadata["source"] = file_path

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n\n\n", "\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    print(f"  → {file_path}: {len(chunks)} chunks")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectors = []
    for i, chunk in enumerate(chunks):
        embedding = embeddings.embed_query(chunk.page_content)
        vector_id = (
            f"{doc_type}_{i}_"
            f"{hash(chunk.page_content) % 100000}"
        )
        vectors.append({
            "id": vector_id,
            "values": embedding,
            "metadata": {
                "text": chunk.page_content,
                "type": doc_type,
                "source": file_path
            }
        })

    batch_size = 50
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch)

    print(f"  ✅ Ingested {file_path} as type='{doc_type}'")
    return len(chunks)


if __name__ == "__main__":
    print("\n📥 Connecting to Pinecone...\n")
    index = get_or_create_index()

    print("🗑️  Clearing existing vectors...")
    try:
        index.delete(delete_all=True)
        print("✅ Cleared")
    except Exception as e:
        print(f"⚠️  Could not clear: {e}")

    print("\n📥 Starting ingestion...\n")

    files = [
        ("data/services.txt",                  "services"),
        ("data/pricing.txt",                   "pricing"),
        ("data/faq.txt",                       "faq"),
        ("data/case_studies.txt",              "case"),
        ("data/agency.txt",                    "agency"),
        ("data/facebook_ads_service.txt",      "services"),
        ("data/client_onboarding_process.txt", "onboarding"),
    ]

    total_chunks = 0
    failed = []

    for file_path, doc_type in files:
        if os.path.exists(file_path):
            try:
                count = ingest(file_path, doc_type, index)
                total_chunks += count
            except Exception as e:
                print(f"  ❌ Failed {file_path}: {e}")
                failed.append(file_path)
        else:
            print(f"  ⚠️  Not found: {file_path}")
            failed.append(file_path)

    print(f"\n{'='*50}")
    print(f"✅ Ingestion complete — {total_chunks} chunks")
    print(f"   Files: {len(files)-len(failed)}/{len(files)}")
    if failed:
        print(f"   ⚠️  Failed: {', '.join(failed)}")
    print(f"{'='*50}\n")

    stats = index.describe_index_stats()
    print(f"📊 Pinecone: {stats.total_vector_count} vectors stored")