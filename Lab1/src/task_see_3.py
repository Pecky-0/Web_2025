import os
import pickle
from collections import defaultdict

def print_inverted_index(pkl_file_path, output_file_path):
    """
    从pkl文件读取倒排索引并打印到文本文件
    
    Args:
        pkl_file_path: 倒排索引pkl文件路径
        output_file_path: 输出文本文件路径
    """
    # 加载倒排索引
    print(f"正在加载倒排索引: {pkl_file_path}")
    try:
        with open(pkl_file_path, 'rb') as f:
            data = pickle.load(f)
        
        # 转换为defaultdict以便保持原有结构
        inverted_index = defaultdict(list, data['inverted_index'])
        doc_lengths = data['doc_lengths']
        total_docs = data['total_docs']
        skip_pointers_added = data.get('skip_pointers_added', False)
        
        # 写入到文本文件
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write("倒排索引内容\n")
            f.write("=" * 80 + "\n\n")
            
            # 基本信息
            f.write("基本信息:\n")
            f.write("-" * 40 + "\n")
            f.write(f"总文档数: {total_docs}\n")
            f.write(f"总词项数: {len(inverted_index)}\n")
            f.write(f"是否包含跳表指针: {'是' if skip_pointers_added else '否'}\n\n")
            
            # 倒排表内容
            f.write("倒排表详细内容:\n")
            f.write("-" * 40 + "\n")
            
            # 按词项排序输出
            for term in sorted(inverted_index.keys()):
                doc_list = inverted_index[term]
                f.write(f"词项: {term}\n")
                f.write(f"包含文档数量: {len(doc_list)}\n")
                f.write("文档列表:\n")
                
                # 根据不同的结构类型显示内容
                if doc_list:
                    if isinstance(doc_list[0], tuple):
                        # 基本倒排索引结构 (doc_id, freq)
                        for i, (doc_id, freq) in enumerate(doc_list[:10]):  # 只显示前10个文档
                            f.write(f"  {i+1}. 文档ID: {doc_id}, 频率: {freq}\n")
                        if len(doc_list) > 10:
                            f.write(f"  ... 还有 {len(doc_list)-10} 个文档未显示\n")
                    
                    elif isinstance(doc_list[0], dict):
                        # 单层跳表结构
                        for i, item in enumerate(doc_list[:10]):
                            skip_ptr = item['skip_ptr'] if item['skip_ptr'] is not None else "无"
                            f.write(f"  {i+1}. 文档ID: {item['doc_id']}, 频率: {item['freq']}, 跳表指针: {skip_ptr}\n")
                        if len(doc_list) > 10:
                            f.write(f"  ... 还有 {len(doc_list)-10} 个文档未显示\n")
                    
                    elif isinstance(doc_list[0], list):
                        # 多层跳表结构
                        f.write(f"  跳表层数: {len(doc_list)}\n")
                        for level, level_data in enumerate(doc_list):
                            f.write(f"  第{level+1}层: {len(level_data)}个文档\n")
                            for i, item in enumerate(level_data[:5]):  # 每层显示前5个
                                next_ptr = item.get('next_level_ptr', "无")
                                f.write(f"    {i+1}. 文档ID: {item['doc_id']}, 频率: {item['freq']}, 下一层指针: {next_ptr}\n")
                            if len(level_data) > 5:
                                f.write(f"    ... 还有 {len(level_data)-5} 个文档未显示\n")
                
                f.write("-" * 40 + "\n\n")
        
        print(f"倒排索引已成功导出到: {output_file_path}")
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 {pkl_file_path}")
    except Exception as e:
        print(f"处理过程中出错: {str(e)}")

def main():
    # 配置路径（根据实际情况修改）
    task3_output_path = r'D:\web\lab1\Web_2025\outputs\Task_3'
    pkl_file = os.path.join(task3_output_path, 'inverted_index.pkl')
    output_file = os.path.join(task3_output_path, 'inverted_indexpkl.txt')
    
    # 执行导出
    print_inverted_index(pkl_file, output_file)

if __name__ == "__main__":
    main()