import os
import random

class DataClassifier():

    def __init__(self):
        print("Successfully create DataClassifier!")

    def set_path(self, path, ratio=[0.8, 0.1, 0.1]):
        self.path = path
        self.data_set = []
        self.ratio = ratio

        print("dc init successfully")

    def load_data(self):
        with open(self.path + "/data.txt", 'r') as datas:
            print("Load data successfully!")

            for data in datas:
                triplet = data.split(' ')

                self.data_set.append([triplet[0], triplet[1], triplet[2]])

    def shuffle_data(self):
        shuffled = self.data_set.copy()  # 创建副本以避免修改原始数据
        random.shuffle(shuffled)
        return shuffled
    
    def shuffle_split(self):
        """
        打乱并划分数据集
        
        Args:
            data: 原始数据
            ratios: 训练集、验证集、测试集的比例
            
        Returns:
            tuple: (train_data, val_data, test_data)
        """

        ratios = self.ratio
        
        # 1. 打乱数据
        shuffled_data = self.shuffle_data()
        
        # 2. 计算划分点
        n_total = len(shuffled_data)
        n_train = int(n_total * ratios[0])
        n_val = int(n_total * ratios[1])
        # n_test = n_total - n_train - n_val
        
        # 3. 划分数据集
        train_data = shuffled_data[:n_train]
        val_data = shuffled_data[n_train:n_train + n_val]
        test_data = shuffled_data[n_train + n_val:]
        
        return train_data, val_data, test_data
    
    def save_data(self):
        train_data, val_data, test_data = self.shuffle_split()
        with open(self.path + "/freebase/kg_train.txt", 'w', encoding='utf-8') as t:
            for data in train_data:
                t.write(f"{data[0]} {data[1]} {data[2]}")

        with open(self.path + "/freebase/kg_valid.txt", 'w', encoding='utf-8') as v:
            for data in val_data:
                v.write(f"{data[0]} {data[1]} {data[2]}")

        with open(self.path + "/freebase/kg_test.txt", 'w', encoding='utf-8') as t:
            for data in test_data:
                t.write(f"{data[0]} {data[1]} {data[2]}")