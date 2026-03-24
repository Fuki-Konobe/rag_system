from app.rag.vectorstore import VectorStoreManager
from app.rag.generator import RAGGenerator

def main():
    vdb_manager = VectorStoreManager()
    retriever = vdb_manager.get_retriever()
    
    generator = RAGGenerator()
    chain = generator.get_chain(retriever)

    query = "井出さんが考えるすだち産業の危険とは？"
    
    print(f"質問: {query}\n回答中...")
    response = chain.invoke(query)
    
    print("-" * 30)
    print(response)

if __name__ == "__main__":
    main()