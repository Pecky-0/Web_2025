import os
import gzip

class DataParser():

    def __init__(self):
        print("Successfully create DataParser!")

    def set_path(self, load_path, save_path, pre_fix, data_count=5000):
        self.save_path = save_path
        self.load_path = load_path
        self.pre_fix = pre_fix
        self.data_counts = data_count

        self.str_to_cnt = {}

        self.str_ptr = 0 # 字符串映射数字
        self.str_to_idx = {}

        print("dp init successfully")

    def get_str_cnt(self, str):
        if str in self.str_to_cnt:
            self.str_to_cnt[str] += 1
        else:
            self.str_to_cnt[str] = 1

        return self.str_to_cnt[str]
        
    def get_str_idx(self, str):
        if str in self.str_to_idx:
            idx = self.str_to_idx[str]
            return idx
        
        idx = self.str_ptr
        self.str_to_idx[str] = idx
        self.str_ptr += 1
        return idx
    
    def remove_prefix(self, text):
        """如果字符串以指定前缀开头，则去除前缀"""
        return text[len(self.pre_fix):]

    def remap_data(self):
        """
        对数据进行重映射 -> 0, 1, 2
        """
        write_datas = []

        cnt = 0

        with gzip.open(self.load_path + "/freebase_douban.gz", 'rb') as datas:
            print("Load data successfully!")

            for data in datas:
                data = data.strip()
                triplet = data.decode().split('\t')

                # print(triplet)

                if triplet[0].startswith(self.pre_fix) and triplet[1].startswith(self.pre_fix) and triplet[2].startswith(self.pre_fix):
                    str0 = self.remove_prefix(triplet[0])
                    str1 = self.remove_prefix(triplet[1])
                    str2 = self.remove_prefix(triplet[2])

                    cnt0 = self.get_str_cnt(str0)
                    cnt1 = self.get_str_cnt(str1)
                    cnt2 = self.get_str_cnt(str2)

                    if cnt0 < 10 or cnt1 < 10 or cnt2 < 10:
                        continue

                    id0 = self.get_str_idx(str0)
                    id1 = self.get_str_idx(str1)
                    id2 = self.get_str_idx(str2)

                    write_datas.append([id0, id1, id2])

                    cnt += 1
                    if cnt >= self.data_counts:
                        break

        return write_datas


            
    def save_data(self):
        save_datas = self.remap_data()

        with open(self.save_path + "/data.txt", 'w', encoding='utf-8') as f:
            for data in save_datas:
                f.write(f"{data[0]} {data[1]} {data[2]}\n")