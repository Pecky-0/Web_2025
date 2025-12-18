# Lab3

## Pecky Add:

在运行前，请记得在自己的python环境里执行：

````
pip install -r requirements.txt
````

以及我想说，这langchain的版本管理就是**一坨屎**

（install了langchain之后还会缺一堆东西需要安，所以我直接把我的版本保存下来了，你们直接安这个环境吧）

为了防止每次都从远处拉一个model下来，先在Lab3目录下执行：

````
python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('moka-ai/m3e-base'); model.save('./model')"
````

这样可以加快一点，代码里是直接用相对路径调用的



FAISS数据库的数据因为占的不是很大，所以我就一块push上来了，如果想在本地跑的话，看main.py里注释掉的部分



说实话，跑出来的结果非常的狗屎，不知道是不是我写的问题：

​	如果是我写的问题的话，问题可能出在modules/faiss_store.py那个文件里，你们可以查一查

反正跑是能跑通了，加油（（（
