import os
import pickle
from typing import List, Optional
import numpy as np
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self, model, model_name='moka-ai/m3e-base'):
        self.model_name = model_name
        self.model = model
        self.vector_store = None

    def save_index(self, path="./faiss_index"):
        """保存索引到磁盘"""
        if self.vector_store:
            self.vector_store.save_local(path)
            print(f"索引已保存到 {path}")

    def get_retriever(self):
        return self.vector_store.as_retriever()
    
    def create_from_documents(self, docs, datas,  metadatas, embeddings, index_path):
        """
        从文档创建 FAISS 索引
        
        Args:
            documents: 文档列表
            metadata: 元数据列表
            index_path: 索引保存路径
        """
        
        # 创建 FAISS 向量存储
        self.vector_store = FAISS.from_embeddings(
            text_embeddings=list(zip(datas, embeddings)),
            embedding=self._get_langchain_embedding(),
            metadatas=metadatas
        )
        
        # 保存索引
        self.save_index(index_path)
        
        print(f"FAISS 索引创建完成，包含 {len(docs)} 个文档")
        return self.vector_store
    
    def _get_langchain_embedding(self):
        """获取 LangChain 兼容的嵌入函数"""
        from langchain_huggingface import HuggingFaceEmbeddings
        
        return HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    
    def load_index(self, path="./faiss_index"):
        """从磁盘加载索引"""
        from langchain_huggingface import HuggingFaceEmbeddings
        
        embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.vector_store = FAISS.load_local(
            folder_path=path,
            embeddings=embeddings,
            allow_dangerous_deserialization=True  # 注意安全警告
        )
        print(f"从 {path} 加载索引完成")
        return self.vector_store
    
    def similarity_search(self, 
                         query, 
                         k=5,
                         score_threshold=0.5):
        """
        相似度搜索
        
        Args:
            query: 查询文本
            k: 返回结果数量
            score_threshold: 相似度阈值
            
        Returns:
            相似文档列表
        """
        if not self.vector_store:
            raise ValueError("请先创建或加载索引")
        
        # # 使用相似度搜索
        # results = self.vector_store.similarity_search_with_relevance_scores(
        #     query, 
        #     k=k
        # )
        
        # # 过滤低于阈值的文档
        # filtered_results = [
        #     (doc, score) for doc, score in results 
        #     if score >= score_threshold
        # ]

        results = self.vector_store.similarity_search(query, k=k)
        
        return results
    
    def add_documents(self, documents: List[str], metadata: Optional[List[dict]] = None):
        """向现有索引添加文档"""
        if not self.vector_store:
            raise ValueError("请先创建或加载索引")
        
        if metadata is None:
            metadata = [{} for _ in range(len(documents))]
        
        self.vector_store.add_texts(
            texts=documents,
            metadatas=metadata
        )
        
        print(f"成功添加 {len(documents)} 个新文档到索引")