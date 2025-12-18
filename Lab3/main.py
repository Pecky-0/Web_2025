from modules.data_parser import DataParser
from modules.data_loader import DataLoader
from modules.faiss_store import VectorStore
from modules.llm import LLMManager
from langchain_core.prompts import PromptTemplate


def main():
    data_path = "./data/law_data_3k.csv"
    save_path = "./data/database/faiss_index"
    question_path = "./data/questions"
    answer_path = "./data/answers"

    api_key = "sk-c539fc5b0287438c83fe9fdcb427bdda"
    batch_size = 100
    parser = DataParser(data_path=data_path, save_path=save_path, batch_size=batch_size)

    template = PromptTemplate(
        input_variables=["context", "question"],
        template="""你是专业的法律知识问答助手。你需要使用以下检索到的上下
            文片段来回答问题，禁止根据常识和已知信息回答问题。如果你不知道答案，
            直接回答“未找到相关答案”。
            Question: {question} 
            Context: {context} 
            Answer:
            """
    )

    model = parser.get_model()

    storer = VectorStore(model)

    # 跑起来还挺慢的，所以存起来了
    # parser.import_data()
    # parser.split()
    # parser.vectorize()
    # parser.store_index(storer=storer)

    vectors = storer.load_index(path=save_path)

    loader = DataLoader(question_path, save_path=answer_path)

    llm = LLMManager(api_key)

    retriever = storer.get_retriever()

    llm.set_the_llm(retriever=retriever, prompt=template)

    for i in range (1, 7):
        question = loader.load_question(index=i)
        print(question)

        answer1 = storer.similarity_search(query=question, k=1)
        retriever = answer1[0].page_content
        
        loader.save_answer(answer1[0].page_content, "FAISS", i)
        print(answer1[0].page_content)

        answer2 = llm.get_answer(question=question)
        loader.save_answer(answer2, "LLM", i)
        print(answer2)
        

if __name__ == '__main__':
    main()