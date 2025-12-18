import xml.etree.ElementTree as ET
import json
import os
import re
import string
from datetime import datetime
from collections import defaultdict
from html import unescape

# 添加停用词列表（示例，可以根据需要扩展）
STOP_WORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 
    'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
    'this', 'that', 'these', 'those', 'it', 'its', 'they', 'them', 'their'
}

class TextProcessor:
    """文本处理类，负责分词和规范化"""
    
    def __init__(self):
        self.stop_words = STOP_WORDS
    
    def clean_html_tags(self, text):
        """清除HTML标签"""
        if not text:
            return ""
        
        # 方法1: 使用正则表达式移除HTML标签
        clean_text = re.sub(r'<[^>]+>', ' ', text)
        
        # 方法2: 处理HTML实体（如 &nbsp; &amp; 等）
        clean_text = unescape(clean_text)
        
        # 替换多个连续空格为单个空格
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        return clean_text
    
    def tokenize(self, text):
        """分词处理"""
        if not text:
            return []
        
        # 首先清除HTML标签
        text = self.clean_html_tags(text)
        
        # 将文本转换为小写
        text = text.lower()
        
        # 简单的分词：按空格和标点分割
        tokens = re.findall(r'\b\w+\b', text)
        
        return tokens
    
    def remove_stopwords(self, tokens):
        """去除停用词"""
        return [token for token in tokens if token not in self.stop_words]
    
    def remove_punctuation_numbers(self, tokens):
        """去除标点符号和纯数字"""
        cleaned_tokens = []
        for token in tokens:
            # 去除纯数字
            if token.isdigit():
                continue
            # 去除包含特殊字符的token
            if re.match(r'^[a-zA-Z]+$', token):
                cleaned_tokens.append(token)
        return cleaned_tokens
    
    def stem_words(self, tokens):
        """简单的词干提取（可以使用更复杂的库如nltk）"""
        # 这里实现一个简单的规则化词干提取
        stemmed_tokens = []
        for token in tokens:
            # 简单的规则：去除常见的词尾
            if len(token) > 3:
                if token.endswith('ing'):
                    stemmed_tokens.append(token[:-3])
                elif token.endswith('ed'):
                    stemmed_tokens.append(token[:-2])
                elif token.endswith('s'):
                    stemmed_tokens.append(token[:-1])
                else:
                    stemmed_tokens.append(token)
            else:
                stemmed_tokens.append(token)
        return stemmed_tokens
    
    def normalize_text(self, text):
        """完整的文本规范化流程"""
        # 分词
        tokens = self.tokenize(text)
        # 去除停用词
        tokens = self.remove_stopwords(tokens)
        # 去除标点符号和数字
        tokens = self.remove_punctuation_numbers(tokens)
        # 词干提取
        tokens = self.stem_words(tokens)
        
        return tokens

def parse_event_xml(file_path):
    """解析Event.xml文件 - 增强版，特别关注description"""
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # 提取description并清理HTML标签
        description_elem = root.find('description')
        raw_description = description_elem.text if description_elem is not None else ""
        
        # 初始化文本处理器来清理HTML
        text_processor = TextProcessor()
        clean_description = text_processor.clean_html_tags(raw_description)
        
        event_data = {
            'event_id': root.find('id').text if root.find('id') is not None else None,
            'event_name': root.find('name').text if root.find('name') is not None else None,
            'status': root.find('status').text if root.find('status') is not None else None,
            'description_raw': raw_description,  # 保留原始description用于调试
            'description_clean': clean_description,  # 清理后的description
            'event_url': root.find('event_url').text if root.find('event_url') is not None else None,
            'created_time': convert_timestamp(root.find('created').text) if root.find('created') is not None else None,
            'updated_time': convert_timestamp(root.find('updated').text) if root.find('updated') is not None else None,
            'event_time': convert_timestamp(root.find('time').text) if root.find('time') is not None else None,
            'yes_rsvp_count': root.find('yes_rsvp_count').text if root.find('yes_rsvp_count') is not None else None,
            'maybe_rsvp_count': root.find('maybe_rsvp_count').text if root.find('maybe_rsvp_count') is not None else None,
            'waitlist_count': root.find('waitlist_count').text if root.find('waitlist_count') is not None else None,
            'headcount': root.find('headcount').text if root.find('headcount') is not None else None,
            'visibility': root.find('visibility').text if root.find('visibility') is not None else None,
            'utc_offset': root.find('utc_offset').text if root.find('utc_offset') is not None else None
        }
        
        # 处理rating信息
        rating_elem = root.find('rating')
        if rating_elem is not None:
            event_data['rating_average'] = rating_elem.find('average').text if rating_elem.find('average') is not None else None
            event_data['rating_count'] = rating_elem.find('count').text if rating_elem.find('count') is not None else None
        
        # 处理event hosts信息
        event_hosts = []
        hosts_elem = root.find('event_hosts')
        if hosts_elem is not None:
            for host in hosts_elem.findall('event_hosts_item'):
                host_data = {
                    'member_name': host.find('member_name').text if host.find('member_name') is not None else None,
                    'member_id': host.find('member_id').text if host.find('member_id') is not None else None
                }
                event_hosts.append(host_data)
        event_data['event_hosts'] = event_hosts
        
        # 处理group信息
        group_elem = root.find('group')
        if group_elem is not None:
            event_data['group_info'] = {
                'group_id': group_elem.find('id').text if group_elem.find('id') is not None else None,
                'group_name': group_elem.find('name').text if group_elem.find('name') is not None else None,
                'group_who': group_elem.find('who').text if group_elem.find('who') is not None else None,
                'urlname': group_elem.find('urlname').text if group_elem.find('urlname') is not None else None,
                'join_mode': group_elem.find('join_mode').text if group_elem.find('join_mode') is not None else None,
                'group_lat': group_elem.find('group_lat').text if group_elem.find('group_lat') is not None else None,
                'group_lon': group_elem.find('group_lon').text if group_elem.find('group_lon') is not None else None
            }
        
        # 新增：生成待检索文档（合并所有文本内容）
        event_data['document_text'] = generate_document_text(event_data)
        
        return event_data
    except Exception as e:
        print(f"解析XML文件时出错: {file_path}, 错误: {e}")
        return None

def generate_document_text(data):
    """生成待检索的文档文本 - 合并所有相关文本字段"""
    text_parts = []
    
    # 添加各种文本字段
    if data.get('event_name'):
        text_parts.append(data['event_name'])
    if data.get('description_clean'):  # 使用清理后的description
        text_parts.append(data['description_clean'])
    if data.get('status'):
        text_parts.append(data['status'])
    
    # 添加group信息中的文本
    group_info = data.get('group_info', {})
    if group_info.get('group_name'):
        text_parts.append(group_info['group_name'])
    if group_info.get('group_who'):
        text_parts.append(group_info['group_who'])
    
    # 添加host信息
    for host in data.get('event_hosts', []):
        if host.get('member_name'):
            text_parts.append(host['member_name'])
    
    return ' '.join(text_parts)

def save_normalized_terms(terms, filename):
    """保存规范化后的词项"""
    with open(filename, 'w', encoding='utf-8') as f:
        for i, term in enumerate(terms, 1):
            f.write(f"{term}\n")

def save_document_analysis(original_text, tokens, normalized_terms, filename):
    """保存文档分析结果"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("文档分析报告\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("原始文本:\n")
        f.write("-" * 30 + "\n")
        f.write(original_text[:] + "...\n\n")
        
        f.write(f"分词结果 (共{len(tokens)}个):\n")
        f.write("-" * 30 + "\n")
        f.write(", ".join(tokens[:]))
        if len(tokens) > 50:
            f.write("...\n\n")
        else:
            f.write("\n\n")
        
        f.write(f"规范化词项 (共{len(normalized_terms)}个):\n")
        f.write("-" * 30 + "\n")
        for i, term in enumerate(normalized_terms[:], 1):
            f.write(f"{i:2d}. {term}\n")

def save_raw_description_comparison(original, cleaned, filename):
    """保存原始和清理后description的对比"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("Description清理前后对比\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("原始Description (包含HTML标签):\n")
        f.write("-" * 40 + "\n")
        f.write(original[:] + "...\n\n")
        
        f.write("清理后的Description:\n")
        f.write("-" * 40 + "\n")
        f.write(cleaned[:] + "...\n\n")

def main():
    """主函数 - 修改版，加入文本处理"""
    #file_path = '../../Meetup/All_Unpack'
    file_path = 'D:\web\lab1\Web_2025\Meetup\All_Unpack'
    save_path = 'D:\web\lab1\Web_2025\outputs\Task_2'
    
    # 创建输出目录
    os.makedirs(os.path.join(save_path, 'TXT'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'Normalized_Terms'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'Analysis'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'Description_Comparison'), exist_ok=True)
    
    # 初始化文本处理器
    text_processor = TextProcessor()
    
    count_event = 0
    count_group = 0
    count_member = 0
    processed_count = 0

    # 检查源目录是否存在
    if not os.path.exists(file_path):
        print(f"错误: 源目录不存在: {file_path}")
        return
    
    # 检查目录中是否有文件
    files = os.listdir(file_path)
    if not files:
        print(f"错误: 目录为空: {file_path}")
        return
    
    print(f"找到 {len(files)} 个文件在目录中")

    for file in os.listdir(file_path):
        f_path = os.path.join(file_path, file)
        try:
            # 解析XML文件
            print(f"正在解析文件: {file}")
            
            if file.startswith('PastEvent'):
                data = parse_event_xml(f_path)
                
                if data is None:
                    print(f"  跳过文件（解析失败）: {file}")
                    continue
                    
                count_event += 1
                processed_count += 1
                
                # 保存原始数据
                txt_filename = os.path.join(save_path, 'TXT', f'event_{count_event}.txt')
                save_to_csv_format(data, txt_filename)
                
                # 保存description清理对比
                comparison_filename = os.path.join(save_path, 'Description_Comparison', f'event_{count_event}_comparison.txt')
                save_raw_description_comparison(
                    data.get('description_raw', ''),
                    data.get('description_clean', ''),
                    comparison_filename
                )
                
                # 文本处理和分析
                document_text = data.get('document_text', '')
                if document_text:
                    # 分词和规范化
                    tokens = text_processor.tokenize(document_text)
                    normalized_terms = text_processor.normalize_text(document_text)
                    
                    # 保存规范化词项
                    terms_filename = os.path.join(save_path, 'Normalized_Terms', f'event_{count_event}_terms.txt')
                    save_normalized_terms(normalized_terms, terms_filename)
                    
                    # 保存分析报告
                    analysis_filename = os.path.join(save_path, 'Analysis', f'event_{count_event}_analysis.txt')
                    save_document_analysis(document_text, tokens, normalized_terms, analysis_filename)
                    
                    print(f"  处理完成: {len(normalized_terms)} 个规范化词项")
                else:
                    print(f"  警告: 文档文本为空")
                
            # 可以取消注释来处理Group和Member文件
            # elif file.startswith('Group'):
            #     data = parse_group_xml(f_path)
            #     count_group += 1
            #     txt_filename = os.path.join(save_path, 'TXT', f'group_{count_group}.txt')
            #     save_to_csv_format(data, txt_filename)
                
            # elif file.startswith('Member'):
            #     data = parse_member_xml(f_path)
            #     count_member += 1
            #     txt_filename = os.path.join(save_path, 'TXT', f'member_{count_member}.txt')
            #     save_to_csv_format(data, txt_filename)
            
        except FileNotFoundError as e:
            print(f"错误: 找不到文件 {e.filename}")
        except ET.ParseError as e:
            print(f"错误: XML解析失败 - {e}")
        except Exception as e:
            print(f"错误: {e}")

    # 输出统计信息
    print("\n" + "="*50)
    print("处理完成统计:")
    print(f"成功处理的事件文件: {count_event} 个")
    print(f"小组文件: {count_group} 个") 
    print(f"成员文件: {count_member} 个")
    print(f"总成功处理文件数: {processed_count} 个")

# 原有的convert_timestamp和save_to_csv_format函数保持不变
def convert_timestamp(timestamp_str):
    """将时间戳转换为可读日期"""
    if timestamp_str:
        try:
            timestamp = int(timestamp_str) / 1000
            return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        except (ValueError, TypeError):
            return timestamp_str
    return None

def save_to_csv_format(data, filename):
    """保存数据到CSV格式的文本文件（简化版）"""
    with open(filename, 'w', encoding='utf-8') as f:
        for key, value in data.items():
            if isinstance(value, list):
                f.write(f"{key}:\n")
                for item in value:
                    if isinstance(item, dict):
                        for subkey, subvalue in item.items():
                            f.write(f"  {subkey}: {subvalue}\n")
                    else:
                        f.write(f"  {item}\n")
            elif isinstance(value, dict):
                f.write(f"{key}:\n")
                for subkey, subvalue in value.items():
                    f.write(f"  {subkey}: {subvalue}\n")
            else:
                f.write(f"{key}: {value}\n")
            f.write("\n")

if __name__ == "__main__":
    main()