from data_parser import DataParser
from data_classifier import DataClassifier

def main():
    load_path = "E:\code\Web_2025_data"
    save_path = "../data"
    pre_fix = "<http://rdf.freebase.com/ns/"
    data_count = 5000

    dp = DataParser()

    dp.set_path(load_path=load_path, save_path=save_path, pre_fix=pre_fix, data_count=data_count)
    dp.save_data()

    ratio = [0.8, 0.1, 0.1]

    path = "../data"
    dc = DataClassifier()
    dc.set_path(path=path, ratio=ratio)
    dc.load_data()
    dc.save_data()

if __name__ == '__main__':
    main()