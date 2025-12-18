# Web_2025
Web信息处理与应用课程作业

## Lab1

已完成Task_2的词法解析，见/src/Task_2/parser.py，在该目录下执行该py即可获得正则化后的输出，见/outputs/Task_2文件夹

Task_2是只完成了对Event文件的Description的词项提取，想提取其他的也可以，挺好改的

Task_3的索引表存到了/outputs/Task_3/inverted_index.pkl，加载方法可以参考index.py的Line 156

src文件夹下的task_see_3.py和task_see_4.py是用来加载python用二进制存储的列表然后打印结果用的

Task_4同样生成的是一个pkl文件

Task_5的实现代码（一个测试对应一个py文件）在src/Task_3下，除了index.py外共五个。

Task_5实验结果在outputs/Task_3/experiments下，共五个.csv文件（由于电脑性能波动，有部分文件是在完成实验报告后又重新测试过，所以数据可能和报告的不太一样，但已经检查基本趋势和结论是大致相同的）后续已对实验报告内的数据进行修改，现在应该已符合。

有一些中间产物也在outputs/Task_3下，但不在experiments下。



如果云盘上有问题的话也可以去Github上找：

### 文件目录

src文件夹下目录为：

````
E:.
│  task_see_3.py
│  task_see_4.py
│
├─Task_2
│      parser.py
│
├─Task_3
│      index.py
│      ptr_change.py
│      search_phrase.py
│      short_first_merge.py
│      tf_idf.py
│      zip_compare_result.py
│
└─Task_4
        enhanced.py
````

- Task2：parser.py 负责Token的解析
- Task3：index.py 负责最基础的倒排表的构建
- Task4：enhanced.py 负责倒排表的拓展和优化
- Task5：放到了Task_3目录下，为除index.py外的五个文件

/outputs文件夹下目录：（注：由于文件数量过多，因此我们每个文件夹下只保留了前三个生成的中间文件）

````
E:.
├─Task_2
│  ├─Analysis
│  │      event_1_analysis.txt
│  │      event_2_analysis.txt
│  │      event_3_analysis.txt
│  │
│  ├─Description_Comparison
│  │      event_1_comparison.txt
│  │      event_2_comparison.txt
│  │      event_3_comparison.txt
│  │
│  ├─Normalized_Terms
│  │      event_1_terms.txt
│  │      event_2_terms.txt
│  │      event_3_terms.txt
│  │
│  └─TXT
│          event_1.txt
│          event_2.txt
│          event_3.txt
│
└─Task_3
    │  index_build_report.txt
    │  index_build_report_var1.txt
    │  index_build_report_var2.txt
    │  inverted_index.pkl
    │  inverted_index_norm_orig.pkl
    │  inverted_index_norm_var1.pkl
    │  inverted_index_norm_var2.pkl
    │  inverted_index_var1.pkl
    │  inverted_index_var2.pkl
    │
    └─experiments
            ptr_change_result.csv
            search_phrase_result.csv
            short_first_merge_result.csv
            tf_idf_result.csv
            zip_compare_result.csv
````

其中Task2文件夹下的相关文件为提取Token时生成的中间文件（具体可见实验报告）

Task3文件夹下.txt文件为生成的相关报告，.pkl文件为python存储的倒排表数组，experiments文件夹下为Task5实验的相关数据。
