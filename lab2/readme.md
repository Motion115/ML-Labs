# ML-Lab2: Classification

**代码文件结构说明**

试验的核心部分

- DL-GNN：（图）深度学习方法
- sklearn-ML：调库实现机器学习方法
- handwritten-ML：手写实现机器学习方法

数据集：corpus中为CoraR数据集，processed是经过预处理的数据集

可视化：visualization文件夹



**Dataset in this project**: A split of train, test and validation according to the suggested split from [CoraR dataset](https://github.com/THUDM/Refined-cora-citeseer). Note that the prediction label(i.e. y) is now a categorical value in the 303th dimension of the feature vector (as apposed to a separate one-hot vector in another file). Extraction of prediction label is required when loading this dataset.

**Visualization**：We utilize GNNLens2, please find the dependencies required in the original repo ([link](https://github.com/dmlc/GNNLens2)).



