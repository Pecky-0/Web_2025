import os
import pickle
from collections import defaultdict, OrderedDict
from typing import List, Dict, Set, Tuple
import time

class InvertedIndex:
    """倒排索引类"""
    
    def __init__(self):
        self.inverted_index = defaultdict(list)  # 词项 -> [文档ID列表]
        self.doc_lengths = {}  # 文档长度（词项数量）
        self.doc_ids = []  # 文档ID列表
        self.term_positions = defaultdict(dict)  # 词项位置信息：词项 -> {文档ID: [位置列表]}
        self.total_docs = 0
        self.skip_pointers_added = False  # 标记是否已添加跳表指针
    
    def build_basic_index(self, terms_dir: str):
        """构建基本的倒排索引"""
        print("开始构建基本倒排索引...")
        
        # 获取所有词项文件
        term_files = [f for f in os.listdir(terms_dir) if f.endswith('_terms.txt')]
        self.total_docs = len(term_files)
        
        for i, term_file in enumerate(term_files):
            doc_id = term_file.replace('_terms.txt', '')  # 如 'event_1'
            
            with open(os.path.join(terms_dir, term_file), 'r', encoding='utf-8') as f:
                terms = [line.strip() for line in f if line.strip()]
            
            # 记录文档长度
            self.doc_lengths[doc_id] = len(terms)
            self.doc_ids.append(doc_id)
            
            # 构建倒排索引
            term_positions = {}  # 记录词项在文档中的位置
            for position, term in enumerate(terms):
                if term not in term_positions:
                    term_positions[term] = []
                term_positions[term].append(position)
                
                # 添加到倒排索引
                if doc_id not in [doc for doc, _ in self.inverted_index[term]]:
                    self.inverted_index[term].append((doc_id, len(term_positions[term])))
            
            # 保存位置信息
            for term, positions in term_positions.items():
                self.term_positions[term][doc_id] = positions
            
            if (i + 1) % 100 == 0:
                print(f"已处理 {i + 1}/{len(term_files)} 个文档")
        
        print(f"基本倒排索引构建完成，共 {len(self.inverted_index)} 个词项")
    
    def add_skip_pointers(self, step_size: int = 3):
        """添加跳表指针"""
        print("开始添加跳表指针...")
        
        for term, doc_list in self.inverted_index.items():
            if len(doc_list) > step_size:
                # 对文档列表排序（按文档ID）
                sorted_docs = sorted(doc_list, key=lambda x: int(x[0].split('_')[1]))
                
                # 添加跳表指针
                new_doc_list = []
                for i, (doc_id, freq) in enumerate(sorted_docs):
                    skip_info = {
                        'doc_id': doc_id,
                        'freq': freq,
                        'skip_ptr': None
                    }
                    
                    # 计算跳表指针位置
                    skip_index = i + step_size
                    if skip_index < len(sorted_docs):
                        skip_info['skip_ptr'] = skip_index
                    
                    new_doc_list.append(skip_info)
                
                self.inverted_index[term] = new_doc_list
        
        self.skip_pointers_added = True
        print("跳表指针添加完成")
    
    def add_multi_level_skip_pointers(self, levels: int = 2):
        print("开始添加多层跳表指针...")
        
        for term, doc_list in self.inverted_index.items():
            # 检查是否已经添加了单层跳表指针
            if self.skip_pointers_added and isinstance(doc_list[0], dict):
                # 如果已经添加了单层跳表指针，提取文档信息
                doc_info_list = []
                for item in doc_list:
                    if isinstance(item, dict):
                        doc_info_list.append((item['doc_id'], item['freq']))
                    else:
                        doc_info_list.append(item)
            else:
                # 基本结构，直接使用
                doc_info_list = doc_list
            
            if len(doc_info_list) > 10:  # 只为较长的列表添加多层指针
                # 确保文档信息是(doc_id, freq)格式
                sorted_docs = []
                for doc_info in doc_info_list:
                    if isinstance(doc_info, tuple) and len(doc_info) == 2:
                        sorted_docs.append(doc_info)
                    elif isinstance(doc_info, dict):
                        sorted_docs.append((doc_info['doc_id'], doc_info['freq']))
                
                # 按文档ID排序
                sorted_docs = sorted(sorted_docs, key=lambda x: int(x[0].split('_')[1]))
                
                # 构建多层跳表
                multi_level_list = []
                for level in range(levels):
                    level_docs = []
                    step = 2 ** level  # 步长指数增长
                    
                    for i in range(0, len(sorted_docs), step):
                        if i < len(sorted_docs):
                            doc_id, freq = sorted_docs[i]
                            level_info = {
                                'doc_id': doc_id,
                                'freq': freq,
                                'level': level
                            }
                            
                            # 添加指向下一层相同位置的指针
                            if level < levels - 1 and i + step < len(sorted_docs):
                                level_info['next_level_ptr'] = i + step
                            
                            level_docs.append(level_info)
                    
                    multi_level_list.append(level_docs)
                
                self.inverted_index[term] = multi_level_list
        
        print(f"多层跳表指针添加完成，共 {levels} 层")
    
    def save_index(self, filename: str):
        """保存倒排索引到文件"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'inverted_index': dict(self.inverted_index),
                'doc_lengths': self.doc_lengths,
                'doc_ids': self.doc_ids,
                'term_positions': dict(self.term_positions),
                'total_docs': self.total_docs,
                'skip_pointers_added': self.skip_pointers_added
            }, f)
        print(f"倒排索引已保存到: {filename}")
    
    def load_index(self, filename: str):
        """从文件加载倒排索引"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.inverted_index = defaultdict(list, data['inverted_index'])
            self.doc_lengths = data['doc_lengths']
            self.doc_ids = data['doc_ids']
            self.term_positions = defaultdict(dict, data['term_positions'])
            self.total_docs = data['total_docs']
            self.skip_pointers_added = data.get('skip_pointers_added', False)
        print(f"倒排索引已从 {filename} 加载")

class IndexBuilder:
    """索引构建器"""
    
    def __init__(self, task2_output_path: str):
        self.task2_path = task2_output_path
        self.terms_dir = os.path.join(task2_output_path, 'Normalized_Terms')
        self.index = InvertedIndex()
    
    def build_complete_index(self):
        """构建完整的索引（包括优化）"""
        # 1. 构建基本倒排索引
        self.index.build_basic_index(self.terms_dir)
        
        # 2. 添加位置信息（已在build_basic_index中完成）
        
        # 3. 添加跳表指针
        self.index.add_skip_pointers(step_size=3)
        
        # 4. 添加多层跳表指针
        self.index.add_multi_level_skip_pointers(levels=2)
    
    def build_basic_index_only(self):
        """只构建基本索引（用于调试）"""
        self.index.build_basic_index(self.terms_dir)
        return self.index
    
    def analyze_index_statistics(self):
        """分析索引统计信息"""
        print("\n=== 索引统计信息 ===")
        print(f"总文档数: {self.index.total_docs}")
        print(f"总词项数: {len(self.index.inverted_index)}")
        
        if self.index.doc_lengths:
            avg_doc_length = sum(self.index.doc_lengths.values()) / len(self.index.doc_lengths)
            print(f"平均文档长度: {avg_doc_length:.2f}")
        
        # 词项频率分布
        term_freqs = []
        for doc_list in self.index.inverted_index.values():
            if doc_list:
                if isinstance(doc_list[0], tuple):
                    term_freqs.append(len(doc_list))
                elif isinstance(doc_list[0], dict):
                    term_freqs.append(len(doc_list))
                elif isinstance(doc_list[0], list):
                    # 多层结构，计算第一层的文档数
                    term_freqs.append(len(doc_list[0]) if doc_list else 0)
        
        if term_freqs:
            print(f"平均倒排列表长度: {sum(term_freqs) / len(term_freqs):.2f}")
            print(f"最大倒排列表长度: {max(term_freqs)}")
            print(f"最小倒排列表长度: {min(term_freqs)}")
        
        # 最常见的词项
        sorted_terms = sorted(self.index.inverted_index.items(), 
                            key=lambda x: len(x[1]) if isinstance(x[1][0], (tuple, dict)) else len(x[1][0]), 
                            reverse=True)[:10]
        print("\n最常见的10个词项:")
        for term, docs in sorted_terms:
            if isinstance(docs[0], (tuple, dict)):
                doc_count = len(docs)
            else:
                doc_count = len(docs[0]) if docs else 0
            print(f"  {term}: 出现在 {doc_count} 个文档中")

def main():
    """主函数"""
    # 配置路径
    task2_output_path = r"C:\Users\11065\Desktop\web_lab1\outputs\Task_2"
    task3_output_path = r"C:\Users\11065\Desktop\web_lab1\outputs\Task_3"
    
    # 创建输出目录
    os.makedirs(task3_output_path, exist_ok=True)
    
    # 检查Task2输出是否存在
    if not os.path.exists(os.path.join(task2_output_path, 'Normalized_Terms')):
        print("错误: 未找到Task2的输出文件，请先运行parser.py")
        return
    
    # 构建索引
    print("开始Task3: 倒排表的构建与优化")
    builder = IndexBuilder(task2_output_path)
    
    try:
        # 构建完整索引
        start_time = time.time()
        builder.build_complete_index()
        end_time = time.time()
        
        print(f"\n索引构建完成，耗时: {end_time - start_time:.2f} 秒")
        
        # 分析统计信息
        builder.analyze_index_statistics()
        
        # 保存索引
        index_file = os.path.join(task3_output_path, 'inverted_index.pkl')
        builder.index.save_index(index_file)
        
        # 生成索引报告
        generate_index_report(builder, task3_output_path)
        
    except Exception as e:
        print(f"构建索引时出错: {e}")
        print("尝试构建基本索引...")
        
        # 如果完整构建失败，尝试只构建基本索引
        builder.build_basic_index_only()
        builder.analyze_index_statistics()
        
        # 保存基本索引
        index_file = os.path.join(task3_output_path, 'basic_inverted_index.pkl')
        builder.index.save_index(index_file)

def generate_index_report(builder: IndexBuilder, output_path: str):
    """生成索引构建报告"""
    report_file = os.path.join(output_path, 'index_build_report.txt')
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("倒排索引构建报告\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("1. 基本统计信息:\n")
        f.write("-" * 30 + "\n")
        f.write(f"总文档数: {builder.index.total_docs}\n")
        f.write(f"总词项数: {len(builder.index.inverted_index)}\n")
        
        if builder.index.doc_lengths:
            avg_doc_length = sum(builder.index.doc_lengths.values()) / len(builder.index.doc_lengths)
            f.write(f"平均文档长度: {avg_doc_length:.2f}\n\n")
        
        f.write("2. 索引优化结果:\n")
        f.write("-" * 30 + "\n")
        
        f.write("3. 倒排索引示例 (前10个词项):\n")
        f.write("-" * 30 + "\n")
        
        # 显示部分倒排索引内容
        sample_terms = list(builder.index.inverted_index.keys())[:10]
        for term in sample_terms:
            doc_list = builder.index.inverted_index[term]
            f.write(f"{term}: ")
            
            # 计算文档数量
            if isinstance(doc_list[0], (tuple, dict)):
                doc_count = len(doc_list)
            else:
                doc_count = len(doc_list[0]) if doc_list else 0
            f.write(f"{doc_count} 个文档\n")
            
            # 显示结构信息
            if doc_list:
                if isinstance(doc_list[0], tuple):
                    f.write("  结构: 基本元组\n")
                elif isinstance(doc_list[0], dict):
                    f.write("  结构: 单层跳表\n")
                elif isinstance(doc_list[0], list):
                    f.write("  结构: 多层跳表\n")
            
            f.write("\n")

if __name__ == "__main__":
    main()