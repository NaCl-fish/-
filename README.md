# 实验环境

环境：Anaconda python3.8

框架：tensorflow2.0和GitHub社区实现的keras-contrib包(https://github.com/keras-team/keras-contrib)

# 数据集

[NaCl-fish/cat-or-dog--dataset: 12500张猫图和12500张狗图，分为训练集20000张和测试集5000张。 简化的数据集， 供同学们学习使用。 (github.com)](https://github.com/NaCl-fish/cat-or-dog--dataset)

# 运行方式

## 训练

```
python training.py
```

## 测试

```
python test.py
```

​	模型的构建实现在model.py，训练实现在training.py，测试实现在test.py，input_data.py实现batch的划分和文件的处理。data存放训练集和测试集，log用于学习好的模型参数的存放，image用于存放训练&测试过程中生成的图表。此次研究模型代码和生成的图表分类放在“各种model”文件夹。

# 实验结果

​	对无norm版本、实现了BN、LN、IN和GN四种已有算法的四个版本以及FNBB和FNBL两种新的改进算法两个版本进行了比较，并对结果进行了分析。

| 各算法最佳结果 | 训练                                           | 测试                                          |
| -------------- | ---------------------------------------------- | --------------------------------------------- |
| 无norm         | ![](.\各种model及其结果\合集\无norm_train.png) | ![](.\各种model及其结果\合集\无norm_test.png) |
| BN             | ![](.\各种model及其结果\合集\BN_train.png)     | ![](.\各种model及其结果\合集\BN_test.png)     |
| LN             | ![](.\各种model及其结果\合集\LN_train.png)     | ![](.\各种model及其结果\合集\LN_test.png)     |
| IN             | ![](.\各种model及其结果\合集\IN_train.png)     | ![](.\各种model及其结果\合集\IN_test.png)     |
| GN             | ![](.\各种model及其结果\合集\GN_train.png)     | ![](.\各种model及其结果\合集\GN_test.png)     |
| FNBB           | ![](.\各种model及其结果\合集\FNBB_train.png)   | ![](.\各种model及其结果\合集\FNBB_test.png)   |
| FNBL           | ![](.\各种model及其结果\合集\FNBL_train.png)   | ![](.\各种model及其结果\合集\FNBL_test.png)   |

（详情及具体分析见技术报告，由于已训练好的模型参数体积过大，不在提交文件中展示，可以联系作者获取或自行训练）
