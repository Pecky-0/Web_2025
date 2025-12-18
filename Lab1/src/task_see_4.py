import pickle
from collections import defaultdict

class InvertedIndexReader:
    def __init__(self):
        self.inverted_index = defaultdict(list)
        self.term_positions = defaultdict(dict)
        self.doc_lengths = {}
        self.doc_ids = []
        self.total_docs = 0
        self.skip_pointers_added = False

    def load_index(self, pkl_file_path):
        """加载倒排索引文件"""
        try:
            with open(pkl_file_path, 'rb') as f:
                data = pickle.load(f)
                self.inverted_index = defaultdict(list, data['inverted_index'])
                self.term_positions = defaultdict(dict, data['term_positions'])
                self.doc_lengths = data['doc_lengths']
                self.doc_ids = data['doc_ids']
                self.total_docs = data['total_docs']
                self.skip_pointers_added = data.get('skip_pointers_added', False)
            print(f"成功加载索引文件: {pkl_file_path}")
            return True
        except Exception as e:
            print(f"加载索引文件失败: {str(e)}")
            return False

    def print_inverted_index(self, output_file_path):
        """将倒排表打印到文本文件"""
        try:
            with open(output_file_path, 'w', encoding='utf-8') as f:
                f.write("倒排索引表\n")
                f.write("=" * 80 + "\n\n")
                
                # 按词项排序输出
                for term in sorted(self.inverted_index.keys()):
                    postings = self.inverted_index[term]
                    f.write(f"词项: {term}\n")
                    f.write(f"倒排列表长度: {len(postings)}\n")
                    f.write("倒排列表内容:\n")
                    
                    # 根据不同数据结构处理输出
                    if self.skip_pointers_added and isinstance(postings[0], dict):
                        # 处理带跳表指针的格式
                        for idx, item in enumerate(postings):
                            skip_ptr = item['skip_ptr'] if item['skip_ptr'] is not None else "None"
                            f.write(f"  位置 {idx}: 文档ID={item['doc_id']}, 词频={item['freq']}, 跳表指针={skip_ptr}\n")
                    elif isinstance(postings[0], list) and len(postings) > 0 and isinstance(postings[0][0], dict):
                        # 处理多层跳表格式
                        for level, level_data in enumerate(postings):
                            f.write(f"  跳表层级 {level}:\n")
                            for item in level_data:
                                next_ptr = item['next_level_ptr'] if item['next_level_ptr'] is not None else "None"
                                f.write(f"    文档ID={item['doc_id']}, 词频={item['freq']}, 下一跳指针={next_ptr}\n")
                    else:
                        # 处理基本格式 (doc_id, freq)
                        for doc_id, freq in postings:
                            f.write(f"  文档ID={doc_id}, 词频={freq}\n")
                    
                    f.write("\n" + "-" * 80 + "\n\n")
                
                f.write(f"索引统计信息:\n")
                f.write(f"总文档数: {self.total_docs}\n")
                f.write(f"总词项数: {len(self.inverted_index)}\n")
                f.write(f"是否包含跳表指针: {'是' if self.skip_pointers_added else '否'}\n")
            
            print(f"倒排表已成功导出到: {output_file_path}")
            return True
        except Exception as e:
            print(f"导出倒排表失败: {str(e)}")
            return False

def main():
    # 配置文件路径
    pkl_file = r'\web\lab1\Web_2025\outputs\Task_4\enhanced_inverted_index.pkl'  # 替换为实际的pkl文件路径
    output_file = '\web\lab1\Web_2025\outputs\Task_4\inverted_indexpkl.txt'  # 输出文本文件路径

    # 读取并导出倒排表
    reader = InvertedIndexReader()
    if reader.load_index(pkl_file):
        reader.print_inverted_index(output_file)

if __name__ == "__main__":
    main()