from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import CharacterTextSplitter
from sentence_transformers import SentenceTransformer

class DataParser():
    def __init__(self, data_path, save_path, batch_size):
        self.data_path = data_path
        self.save_path = save_path
        self.batch_size = batch_size

    def import_data(self):
        self.loader = CSVLoader(self.data_path, encoding="utf-8")
        self.docs = self.loader.load()
        print("Load data successfully")

    def split(self):
        self.splitter = CharacterTextSplitter(
                            chunk_size=500,
                            chunk_overlap=50,
                            separator="\n\n",  # 优先按段落分割
                        )
        
        self.docs = self.splitter.split_documents(self.docs)
        self.datas = [doc.page_content for doc in self.docs]
        self.meta_datas = [doc.metadata for doc in self.docs]

        print("Split data successfully")

        # print(self.datas)

    def vectorize(self):
        # 用的 m3e-base 的模型

        self.model = SentenceTransformer('./model')
        print("Load model successfully")

        all_embeddings = []

        for i in range(0, len(self.datas), self.batch_size):
            batch = self.datas[i:i+self.batch_size]
            batch_embeddings = self.model.encode(batch)
            all_embeddings.extend(batch_embeddings)
            print(f"已编码 {i+len(batch)}/{len(self.datas)} 个文档")

        self.embeddings = all_embeddings

        # for embedding in self.embeddings:
        #     print(embedding)

        print("Vectorize data successfully")

    def store_index(self, storer):
        self.storer = storer
        self.storer.create_from_documents(self.docs, self.datas, self.meta_datas, self.embeddings, self.save_path)

    def get_model(self):
        self.model = SentenceTransformer('./model')
        print("Load model successfully")
        return self.model
