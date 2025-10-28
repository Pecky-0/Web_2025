import os
import pickle
import struct
from collections import defaultdict
from typing import List, Dict, Tuple
import time

# 块大小配置（可根据内存调整）
BLOCK_SIZE = 1024 * 1024  # 1MB per block

class InvertedIndex:
    """增强版倒排索引类，支持位置信息、块存储和前端编码"""
    
    def __init__(self):
        self.inverted_index = defaultdict(list)  # 词项 -> 倒排列表
        self.doc_lengths = {}  # 文档长度（词项数量）
        self.doc_ids = []  # 文档ID列表
        self.term_positions = defaultdict(dict)  # 词项位置: 词项 -> {文档ID: [位置列表]}
        self.total_docs = 0
        self.skip_pointers_added = False  # 跳表标记
        self.blocks = []  # 块存储容器
        self.block_metadata = {}  # 块元数据: {词项: (块索引, 偏移量)}

    def build_basic_index(self, terms_dir: str):
        """构建包含完整位置信息的基本倒排索引"""
        print("开始构建带位置信息的基本倒排索引...")
        
        term_files = [f for f in os.listdir(terms_dir) if f.endswith('_terms.txt')]
        self.total_docs = len(term_files)
        
        for i, term_file in enumerate(term_files):
            doc_id = term_file.replace('_terms.txt', '')  # 如 'event_1'
            
            with open(os.path.join(terms_dir, term_file), 'r', encoding='utf-8') as f:
                terms = [line.strip() for line in f if line.strip()]
            
            # 记录文档长度和ID
            self.doc_lengths[doc_id] = len(terms)
            self.doc_ids.append(doc_id)
            
            # 记录词项在文档中的位置（从1开始计数，便于短语检索）
            term_positions = defaultdict(list)
            for pos, term in enumerate(terms, 1):
                term_positions[term].append(pos)
            
            # 更新倒排索引和位置信息
            for term, positions in term_positions.items():
                # 倒排列表存储 (文档ID, 词频)
                self.inverted_index[term].append((doc_id, len(positions)))
                # 位置信息存储完整位置列表
                self.term_positions[term][doc_id] = positions
            
            if (i + 1) % 100 == 0:
                print(f"已处理 {i + 1}/{len(term_files)} 个文档")
        
        print(f"基本倒排索引构建完成，共 {len(self.inverted_index)} 个词项")

    def add_skip_pointers(self, step_size: int = 3):
        """添加跳表指针（保持原有逻辑）"""
        print("开始添加跳表指针...")
        
        for term, doc_list in self.inverted_index.items():
            if len(doc_list) > step_size:
                # 按文档ID排序
                sorted_docs = sorted(doc_list, key=lambda x: int(x[0].split('_')[1]))
                
                # 添加跳表指针
                new_doc_list = []
                for i, (doc_id, freq) in enumerate(sorted_docs):
                    skip_info = {
                        'doc_id': doc_id,
                        'freq': freq,
                        'skip_ptr': i + step_size if (i + step_size) < len(sorted_docs) else None
                    }
                    new_doc_list.append(skip_info)
                
                self.inverted_index[term] = new_doc_list
        
        self.skip_pointers_added = True
        print("跳表指针添加完成")

    def add_multi_level_skip_pointers(self, levels: int = 2):
        """添加多层跳表指针（保持原有逻辑）"""
        print("开始添加多层跳表指针...")
        
        for term, doc_list in self.inverted_index.items():
            # 提取文档信息
            if self.skip_pointers_added and isinstance(doc_list[0], dict):
                doc_info_list = [(item['doc_id'], item['freq']) for item in doc_list]
            else:
                doc_info_list = doc_list
            
            if len(doc_info_list) > 10:
                # 按文档ID排序
                sorted_docs = sorted(doc_info_list, key=lambda x: int(x[0].split('_')[1]))
                
                # 构建多层跳表
                multi_level_list = []
                for level in range(levels):
                    step = 2 ** level
                    level_docs = []
                    for i in range(0, len(sorted_docs), step):
                        doc_id, freq = sorted_docs[i]
                        level_info = {
                            'doc_id': doc_id,
                            'freq': freq,
                            'level': level,
                            'next_level_ptr': i + step if (i + step) < len(sorted_docs) else None
                        }
                        level_docs.append(level_info)
                    multi_level_list.append(level_docs)
                
                self.inverted_index[term] = multi_level_list
        
        print(f"多层跳表指针添加完成，共 {levels} 层")

    def apply_front_coding(self, term_list: List[str]) -> List[Tuple[str, int, str]]:
        """对词项列表应用前端编码压缩"""
        if not term_list:
            return []
        
        encoded = []
        # 按字典序排序
        sorted_terms = sorted(term_list)
        # 第一个词项不压缩
        first_term = sorted_terms[0]
        encoded.append((first_term, 0, ""))
        
        for i in range(1, len(sorted_terms)):
            prev_term = sorted_terms[i-1]
            curr_term = sorted_terms[i]
            # 计算公共前缀长度
            common_len = 0
            while (common_len < len(prev_term) and 
                   common_len < len(curr_term) and 
                   prev_term[common_len] == curr_term[common_len]):
                common_len += 1
            # 存储(公共前缀长度, 后缀)
            encoded.append((curr_term, common_len, curr_term[common_len:]))
        
        return encoded

    def split_into_blocks(self):
        """将倒排索引按块存储"""
        print("开始按块存储索引...")
        current_block = {}
        current_size = 0
        
        # 对所有词项按字典序排序（便于前端编码）
        sorted_terms = sorted(self.inverted_index.keys())
        # 应用前端编码
        encoded_terms = self.apply_front_coding(sorted_terms)
        
        for term, common_len, suffix in encoded_terms:
            # 计算当前词项索引数据大小（近似）
            term_data = {
                'postings': self.inverted_index[term],
                'positions': self.term_positions[term]
            }
            data_size = len(pickle.dumps(term_data))
            
            # 检查是否超过块大小
            if current_size + data_size > BLOCK_SIZE and current_block:
                # 保存当前块
                self.blocks.append(current_block)
                # 记录块元数据
                for t in current_block:
                    self.block_metadata[t] = (len(self.blocks)-1, len(pickle.dumps(current_block[t])))
                # 重置当前块
                current_block = {}
                current_size = 0
            
            # 添加到当前块（存储编码后的信息）
            current_block[term] = {
                'encoded': (common_len, suffix),
                'postings': self.inverted_index[term],
                'positions': self.term_positions[term]
            }
            current_size += data_size
        
        # 保存最后一个块
        if current_block:
            self.blocks.append(current_block)
            for t in current_block:
                self.block_metadata[t] = (len(self.blocks)-1, len(pickle.dumps(current_block[t])))
        
        print(f"块存储完成，共 {len(self.blocks)} 个块")

    def save_index(self, filename: str):
        """保存索引（包含块存储和编码信息）"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'inverted_index': dict(self.inverted_index),
                'term_positions': dict(self.term_positions),
                'doc_lengths': self.doc_lengths,
                'doc_ids': self.doc_ids,
                'total_docs': self.total_docs,
                'skip_pointers_added': self.skip_pointers_added,
                'blocks': self.blocks,
                'block_metadata': self.block_metadata
            }, f)
        print(f"增强版倒排索引已保存到: {filename}")

    def load_index(self, filename: str):
        """加载索引"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.inverted_index = defaultdict(list, data['inverted_index'])
            self.term_positions = defaultdict(dict, data['term_positions'])
            self.doc_lengths = data['doc_lengths']
            self.doc_ids = data['doc_ids']
            self.total_docs = data['total_docs']
            self.skip_pointers_added = data.get('skip_pointers_added', False)
            self.blocks = data.get('blocks', [])
            self.block_metadata = data.get('block_metadata', {})
        print(f"增强版倒排索引已从 {filename} 加载")

    def search_phrase(self, phrase: List[str]) -> List[str]:
        """短语检索实现（利用位置信息）"""
        if len(phrase) < 2:
            return []
        
        # 获取第一个词项的文档列表
        if phrase[0] not in self.term_positions:
            return []
        candidate_docs = set(self.term_positions[phrase[0]].keys())
        
        # 检查其他词项是否在候选文档中
        for term in phrase[1:]:
            if term not in self.term_positions:
                return []
            candidate_docs.intersection_update(self.term_positions[term].keys())
            if not candidate_docs:
                return []
        
        # 验证位置连续性
        result_docs = []
        for doc_id in candidate_docs:
            positions_list = [self.term_positions[term][doc_id] for term in phrase]
            # 检查是否存在连续位置序列
            for pos in positions_list[0]:
                match = True
                for i in range(1, len(phrase)):
                    if (pos + i) not in positions_list[i]:
                        match = False
                        break
                if match:
                    result_docs.append(doc_id)
                    break
        
        return result_docs


class IndexBuilder:
    """增强版索引构建器"""
    
    def __init__(self, task2_output_path: str):
        self.task2_path = task2_output_path
        self.terms_dir = os.path.join(task2_output_path, 'Normalized_Terms')
        self.index = InvertedIndex()
    
    def build_complete_index(self):
        """构建完整增强版索引"""
        # 1. 构建带位置信息的基本索引
        self.index.build_basic_index(self.terms_dir)
        
        # 2. 添加跳表优化
        self.index.add_skip_pointers(step_size=3)
        self.index.add_multi_level_skip_pointers(levels=2)
        
        # 3. 应用块存储和前端编码
        self.index.split_into_blocks()
    
    def build_basic_index_only(self):
        """只构建基本索引（用于调试）"""
        self.index.build_basic_index(self.terms_dir)
        return self.index
    
    def analyze_index_statistics(self):
        """分析索引统计信息（增强版）"""
        print("\n=== 增强版索引统计信息 ===")
        print(f"总文档数: {self.index.total_docs}")
        print(f"总词项数: {len(self.index.inverted_index)}")
        print(f"块数量: {len(self.index.blocks)}")
        
        if self.index.doc_lengths:
            avg_doc_length = sum(self.index.doc_lengths.values()) / len(self.index.doc_lengths)
            print(f"平均文档长度: {avg_doc_length:.2f}")
        
        # 位置信息统计
        total_positions = sum(len(pos) for term_pos in self.index.term_positions.values() 
                             for pos in term_pos.values())
        print(f"总位置信息条目数: {total_positions}")


def main():
    """主函数"""
    # 配置路径
    task2_output_path = r'\web\lab1\Web_2025\outputs\Task_2'
    task4_output_path = r'\web\lab1\Web_2025\outputs\Task_4'
    
    os.makedirs(task4_output_path, exist_ok=True)
    
    # 检查Task2输出
    if not os.path.exists(os.path.join(task2_output_path, 'Normalized_Terms')):
        print("错误: 未找到Task2的输出文件，请先运行parser.py")
        return
    
    # 构建增强版索引
    print("开始构建增强版倒排索引（带位置信息、块存储和前端编码）")
    builder = IndexBuilder(task2_output_path)
    
    try:
        start_time = time.time()
        builder.build_complete_index()
        end_time = time.time()
        
        print(f"\n索引构建完成，耗时: {end_time - start_time:.2f} 秒")
        
        # 分析统计
        builder.analyze_index_statistics()
        
        # 保存索引
        index_file = os.path.join(task4_output_path, 'enhanced_inverted_index.pkl')
        builder.index.save_index(index_file)
        
        # 生成报告
        generate_index_report(builder, task4_output_path)
        
    except Exception as e:
        print(f"构建索引时出错: {e}")
        # 降级处理
        builder.build_basic_index_only()
        builder.analyze_index_statistics()
        index_file = os.path.join(task4_output_path, 'basic_enhanced_index.pkl')
        builder.index.save_index(index_file)


def generate_index_report(builder: IndexBuilder, output_path: str):
    """生成增强版索引报告"""
    report_file = os.path.join(output_path, 'enhanced_index_report.txt')
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("增强版倒排索引构建报告\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("1. 基本统计信息:\n")
        f.write("-" * 30 + "\n")
        f.write(f"总文档数: {builder.index.total_docs}\n")
        f.write(f"总词项数: {len(builder.index.inverted_index)}\n")
        f.write(f"块数量: {len(builder.index.blocks)}\n")
        
        if builder.index.doc_lengths:
            avg_doc_length = sum(builder.index.doc_lengths.values()) / len(builder.index.doc_lengths)
            f.write(f"平均文档长度: {avg_doc_length:.2f}\n\n")
        
        f.write("2. 增强功能信息:\n")
        f.write("-" * 30 + "\n")
        f.write("• 已添加完整词项位置信息（支持短语检索）\n")
        f.write("• 已应用前端编码压缩\n")
        f.write(f"• 按块存储: 块大小 {BLOCK_SIZE/1024:.0f}KB\n\n")
        
        f.write("3. 倒排索引示例:\n")
        f.write("-" * 30 + "\n")
        sample_terms = list(builder.index.inverted_index.keys())[:5]
        for term in sample_terms:
            f.write(f"词项: {term}\n")
            f.write(f"  倒排列表长度: {len(builder.index.inverted_index[term])}\n")
            f.write(f"  位置信息文档数: {len(builder.index.term_positions[term])}\n\n")


if __name__ == "__main__":
    main()