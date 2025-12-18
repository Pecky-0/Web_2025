

class DataLoader():
    def __init__(self, load_path, save_path):
        self.load_path = load_path
        self.save_path = save_path

    def load_question(self, index):
        file_name = self.load_path + f"/question_{index}.txt"

        with open(file_name, 'r', encoding='utf-8') as f:
            content = f.read()
        return content.strip()  # 去除首尾空白字符
    
    def save_answer(self, text, name, index):
        with open(self.save_path + f"/answer_{name}_{index}.txt", 'w', encoding='utf-8') as f:
            f.write(text)
