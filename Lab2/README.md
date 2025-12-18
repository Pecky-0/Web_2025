# *一些关于数据处理的注意事项

* **说明：以下注意事项仅适用于我们提供的代码框架！！**
  - 如果你选择 **自行实现数据处理与加载流程**，或完全基于 **自建代码** 进行实验：
     **可以忽略本 README**，只要最终完整地跑通实验即可。
  - 如果你选择 **使用我们提供的代码框架**：
     为确保数据载入模块能够正常运行，请务必按照下述方式处理并保存数据。



## Pecky ADD：

文件目录树：

````
│  main_kg.py
│  README.md
│
├─data
│  │  data.txt
│  │
│  └─freebase
│          kg_test.txt
│          kg_train.txt
│          kg_valid.txt
│
├─data_loader
│  │  loader_kg.py
│  │
│  └─__pycache__
│          loader_kg.cpython-313.pyc
│
├─data_parser
│  │  data_classifier.py
│  │  data_parser.py
│  │  main_parser.py
│  │
│  └─__pycache__
│          data_classifier.cpython-313.pyc
│          data_parser.cpython-313.pyc
│
├─model
│  │  KG_embedding_model.py
│  │
│  └─__pycache__
│          KG_embedding_model.cpython-313.pyc
│
├─parser
│  │  parser_Embedding_based.py
│  │
│  └─__pycache__
│          parser_Embedding_based.cpython-313.pyc
│
├─trained_model
│  │  Embedding_based.pth
│  │
│  └─freebase
│
└─utils
    │  log_helper.py
    │  metrics.py
    │  model_helper.py
    │
    └─__pycache__
            log_helper.cpython-313.pyc
            metrics.cpython-313.pyc
            model_helper.cpython-313.pyc
````

- **data_loader**

  - loader_kg.py

    用于加载处理好后的数据，里面貌似有三个TODO

- **model**

  - KG_embedding_model.py

    没仔细看，应该是需要完成的主要内容，有一大堆要填的东西

- **parser**

  - parser_Embedding_based.py

    这是个参数解析器，负责解析运行时传入的参数，看起来比较重要的参数：

    - --seed：随机数种子
    - --cuda：用GPU计算（默认是false），要是显卡是NVIDIA的，可以加上 --cuda true

    如果要改些训练阶段的参数的话，可以考虑下面这些：

    - --embed_dim：
    - --relation_dim：这两个是跟算法有关系的，TransE里好像要保证一致，TransR里貌似不用（**后面再确认一下算法流程**）
      - 这俩默认都是32维的
    - --KG_embedding_type：选用的算法，有两个可选项（TransR，TransE）
    - --kg_l2loss_lambda：计算loss时的参数，默认1e-4
    - --lr：学习率，默认1e-3
    - --n_epoch：训练轮数，默认1000

- **utils**

  看起来是一些辅助功能，比如打印日志，衡量模型好坏的，没有什么需要填的

- **data**

​	下面的data_parser生成的数据文件，我就选了5000个，量不大，所以直接存下来了

- **data_parser**

  这个是我自己写的，相当于Task1跟Task2

  - data_parser.py

    用来把数据集映射成index

  - data_classifier.py

    把数据打乱后分为训练集，验证集跟测试集

  - main_parser.py

    我懒得改框架里的main代码了，所以在这里单开了一个，所以，**如果想要自己重新生成数据（比如打算改改数据规模），记得运行这个main代码**

- main_kg.py

  这是整个主程序的代码，里面有训练阶段和测试阶段（需要在main函数里自己用注释来切换，也可以加个参数控制，后面有空再改吧）

  训练的代码流程大概是：

  ​	从训练集里每次取一小部分的数据（数据规模跟参数args里的batch_size有关系）

  ​	用PyTorch的网络自动根据loss优化

  ​	在训练的时候会根据验证集的命中率啥的指标来确定是不是要保存当前的模型（最后默认会存在/trained_model/free_base/...的文件夹里）

  如果要进行测试阶段看指标的话，需要把你想看的模型**手动拉到/trained_model下改名为Embedding_based.pth**



## 1. 子图构建后的编号重映射

在抽取豆瓣电影知识图谱子图的过程中，需要完成 **编号重映射**：将所有 Freebase 的实体 ID 和关系 ID 映射为从 **0 开始**的连续编号。

注：该步骤可在子图抽取过程中完成，也可在抽取完成后统一处理。

* 重映射形式类似于（仅作为示例）：

  * 实体：
    ```
    0 m.09gb_4p
    1 m.0dgs73j
    2 m.09gq0x5
    3 m.0211pk
    4 m.0bwky98
    5 m.05345qj
    ```

  * 关系
    ```
    0 type.type.instance
    1 film.film_crew_gig.film
    2 film.film_regional_release_date.film
    3 film.performance.film
    4 music.soundtrack.film
    ```

* 基于此，最终三元组中的实体、关系需要以从0开始的编号表示（**建议将重映射后的三元组保存为文件，以便后续数据集划分使用**），即格式类似于：
  ```
  578 0 8
  579 0 142
  579 0 144
  579 0 143
  580 1 23
  ```

  * 其中每一行为一个三元组，如第一行“578 0 8”代表三元组（578，0，8）。578为头实体从0开始的新编号，同理，0为关系编号，8为尾实体编号。



## 2. 数据集划分要求

若使用提供的代码流程，为确保能够正常读取数据，请将划分后的三份数据集命名为：`train.txt`，`valid.txt`，`test.txt`。

要求：

* 每行三元组格式为 `h r t`，以空格分隔。
* 实体与关系编号必须均从 **0 开始**。
* 文件保存至 `data/freebase/` 目录下。

以下为 `train.txt` 的示例：

```
249 43 87587
524 73 46502
229 51 16115
11 130 49260
74673 2 303
```



> 如有任何问题，可向助教确认或反馈
>
> 祝大家实验顺利~