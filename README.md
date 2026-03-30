# Advanced Machine Learning: Logistic Regression
# 高级机器学习：逻辑回归

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-green.svg)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.5+-orange.svg)](https://matplotlib.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-red.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

The project focuses on implementing a Logistic Regression model from scratch to perform binary classification on a toy dataset.

本项目专注于从零实现逻辑回归模型，在玩具数据集上执行二分类任务。

---

## 📂 Project Structure / 项目结构

```
├── logistic_regression.py          # Core implementation of Logistic Regression
│                                   # 逻辑回归算法的核心实现
├── Assignment2_25053692_ChenXi.ipynb  # Jupyter Notebook with experimental results
│                                   # 包含实验结果、数据可视化和性能分析的Jupyter Notebook
├── Train_toydata.txt               # Dataset used for training the model
│                                   # 用于训练模型的数据集
└── Test_toydata.txt                # Dataset used for evaluating the model
                                    # 用于评估模型泛化能力的数据集
```

---

## 🛠️ Features / 功能特点

| English | 中文 |
|---------|------|
| **Scratch Implementation**: Manual implementation of the Sigmoid function, Binary Cross-Entropy Loss, and Gradient Descent optimization | **从零实现**：手动实现Sigmoid函数、二分类交叉熵损失和梯度下降优化 |
| **Model Training**: Iterative weight updates using the training set | **模型训练**：使用训练集进行迭代权重更新 |
| **Performance Evaluation**: Calculation of metrics such as Accuracy and Loss on the test set | **性能评估**：计算测试集上的准确率和损失等指标 |
| **Data Visualization**: Plotting the Decision Boundary and the Learning Curve (Loss vs. Iterations) | **数据可视化**：绘制决策边界和学习曲线（损失 vs. 迭代次数） |

---

## 🛠️ Tech Stack / 技术栈

| Category | Technologies |
|----------|--------------|
| **Language** | Python 3.8+ |
| **Scientific Computing** | NumPy 1.21+ |
| **Visualization** | Matplotlib 3.5+ |
| **Development Environment** | Jupyter Notebook, PyCharm |
| **Version Control** | Git, GitHub |

---

## 🚀 Getting Started / 快速开始

### 1. Prerequisites / 环境要求

- **Python 3.8+** - [Download](https://www.python.org/downloads/)
- **pip** (Python package manager)
- **Git** - [Download](https://git-scm.com/)
- **Conda** (recommended) / Miniconda or Anaconda - [Download](https://docs.conda.io/en/latest/miniconda.html)

---

### 2. Environment Setup / 环境配置

#### Option A: Using Conda (Recommended) / 选项A：使用Conda（推荐）

```bash
# Create a new conda environment with Python 3.10
# 创建Python 3.10的新conda环境
conda create -n logistic_regression python=3.10 -y

# Activate the environment
# 激活环境
conda activate logistic_regression

# Verify environment is active
# 验证环境已激活
conda info --envs
```

#### Option B: Using Virtual Environment (venv) / 选项B：使用虚拟环境

```bash
# Create virtual environment
# 创建虚拟环境
python -m venv venv

# Activate on Windows / Windows激活
venv\Scripts\activate

# Activate on macOS/Linux / macOS/Linux激活
source venv/bin/activate
```

---

### 3. Installation / 安装依赖

```bash
# Install required packages
# 安装所需包
pip install -r requirements.txt
```

**If `requirements.txt` doesn't exist, create it with:**
**如果 `requirements.txt` 不存在，使用以下命令创建：**

```bash
# Create requirements.txt
# 创建依赖文件
pip install numpy matplotlib jupyter
pip freeze > requirements.txt
```

**Example `requirements.txt` content / 依赖文件示例内容：**
```
numpy>=1.21.0
matplotlib>=3.5.0
jupyter>=1.0.0
```

---

### 4. Verify Installation / 验证安装

```bash
# Verify Python version / 验证Python版本
python --version

# Verify NumPy installation / 验证NumPy安装
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"

# Verify Matplotlib installation / 验证Matplotlib安装
python -c "import matplotlib; print(f'Matplotlib version: {matplotlib.__version__}')"
```

**Expected output / 预期输出：**
```
Python 3.10.x
NumPy version: 1.21.x
Matplotlib version: 3.5.x
```

---

### 5. Running the Project / 运行项目

#### Run Training Script / 运行训练脚本

```bash
# Make sure conda environment is activated
# 确保conda环境已激活
conda activate logistic_regression

# Run the logistic regression script
# 运行逻辑回归脚本
python logistic_regression.py
```

**Expected output / 预期输出：**
```
Starting training...
Iteration 0: Loss = 0.6931
Iteration 100: Loss = 0.4523
Iteration 200: Loss = 0.3512
...
Training completed!
Final Accuracy: 0.9523
Decision boundary plotted.
```

#### Run Jupyter Notebook / 运行Jupyter Notebook

```bash
# Launch Jupyter Notebook
# 启动Jupyter Notebook
jupyter notebook Assignment2_25053692_ChenXi.ipynb

# Alternatively, use Jupyter Lab (if installed)
# 或者使用Jupyter Lab（如果已安装）
jupyter lab
```

#### Alternative: Run in Google Colab / 备选方案：在Google Colab中运行

1. Upload `Assignment2_25053692_ChenXi.ipynb` to Google Colab
2. Upload `Train_toydata.txt` and `Test_toydata.txt`
3. Run cells sequentially

---

### 6. Dataset Format / 数据集格式

**Train_toydata.txt & Test_toydata.txt format:**
**训练集和测试集格式：**

```
x1 x2 label
1.2 3.4 0
2.3 4.5 1
...
```

- **x1, x2**: Feature values / 特征值
- **label**: Binary class (0 or 1) / 二分类标签（0或1）

---

## 📊 Expected Results / 预期结果

| Metric / 指标 | Expected Value / 预期值 |
|--------------|------------------------|
| Training Accuracy / 训练准确率 | > 90% |
| Test Accuracy / 测试准确率 | > 85% |
| Final Loss / 最终损失 | < 0.3 |
| Decision Boundary | Linear separator / 线性分隔线 |

---

## 🔧 Customization / 自定义配置

Edit `logistic_regression.py` to modify hyperparameters:
编辑 `logistic_regression.py` 修改超参数：

```python
# Hyperparameters / 超参数
learning_rate = 0.01      # Learning rate / 学习率
num_iterations = 1000     # Number of iterations / 迭代次数
batch_size = 32           # Batch size (if using mini-batch) / 批量大小（如使用小批量）
```

---

## 📈 Visualizations / 可视化

The project generates two types of visualizations:
本项目生成两种可视化图表：

1. **Decision Boundary Plot / 决策边界图**
   - Shows how the model separates two classes
   - 显示模型如何分隔两个类别
   - Red/blue points represent different classes
   - 红/蓝点代表不同类别

2. **Learning Curve / 学习曲线**
   - Loss vs. Iterations
   - 损失 vs. 迭代次数
   - Shows convergence behavior
   - 显示收敛行为

---

## 🧪 Testing Different Scenarios / 测试不同场景

```bash
# Test with different learning rates
# 测试不同学习率
python logistic_regression.py --lr 0.001
python logistic_regression.py --lr 0.01
python logistic_regression.py --lr 0.1

# Test with different iterations
# 测试不同迭代次数
python logistic_regression.py --iter 500
python logistic_regression.py --iter 2000
```

---

## 🐛 Common Issues & Solutions / 常见问题与解决方案

| Issue / 问题 | Solution / 解决方案 |
|--------------|---------------------|
| **ModuleNotFoundError: No module named 'numpy'** | Activate conda environment: `conda activate logistic_regression`<br>Then run: `pip install numpy` |
| **Jupyter notebook not found** | Install Jupyter: `pip install jupyter`<br>Or use: `python -m pip install jupyter` |
| **Dataset file not found** | Ensure `Train_toydata.txt` and `Test_toydata.txt` are in the same directory as the script<br>确保数据文件与脚本在同一目录 |
| **Poor accuracy (< 70%)** | Increase iterations or adjust learning rate<br>增加迭代次数或调整学习率 |
| **Loss not decreasing** | Check if learning rate is too high or too low<br>检查学习率是否过高或过低 |
| **Conda command not found** | Install Miniconda or Anaconda first<br>首先安装Miniconda或Anaconda |

---

## 📝 Algorithm Details / 算法细节

### Mathematical Foundation / 数学基础

**Sigmoid Function / Sigmoid函数：**
```
σ(z) = 1 / (1 + e^(-z))
```

**Binary Cross-Entropy Loss / 二分类交叉熵损失：**
```
L(y, ŷ) = -[y * log(ŷ) + (1-y) * log(1-ŷ)]
```

**Gradient Descent Update / 梯度下降更新：**
```
w = w - α * ∇L(w)
```

Where / 其中：
- `z = w·x + b` (linear combination / 线性组合)
- `ŷ = σ(z)` (predicted probability / 预测概率)
- `α` = learning rate / 学习率

---

## 📁 Complete Setup Commands / 完整设置命令

### For Windows / Windows系统

```bash
# 1. Clone repository / 克隆仓库
git clone https://github.com/your-username/your-repo.git
cd your-repo

# 2. Create and activate conda environment / 创建并激活conda环境
conda create -n logistic_regression python=3.10 -y
conda activate logistic_regression

# 3. Install dependencies / 安装依赖
pip install numpy matplotlib jupyter

# 4. Verify installation / 验证安装
python -c "import numpy, matplotlib; print('Setup successful!')"

# 5. Run training / 运行训练
python logistic_regression.py

# 6. Launch Jupyter (optional) / 启动Jupyter（可选）
jupyter notebook Assignment2_25053692_ChenXi.ipynb

# 7. Deactivate environment when done / 完成后退出环境
conda deactivate
```

### For macOS/Linux / macOS/Linux系统

```bash
# 1. Clone repository / 克隆仓库
git clone https://github.com/your-username/your-repo.git
cd your-repo

# 2. Create and activate conda environment / 创建并激活conda环境
conda create -n logistic_regression python=3.10 -y
conda activate logistic_regression

# 3. Install dependencies / 安装依赖
pip install numpy matplotlib jupyter

# 4. Verify installation / 验证安装
python -c "import numpy, matplotlib; print('Setup successful!')"

# 5. Run training / 运行训练
python logistic_regression.py

# 6. Launch Jupyter (optional) / 启动Jupyter（可选）
jupyter notebook Assignment2_25053692_ChenXi.ipynb

# 7. Deactivate environment when done / 完成后退出环境
conda deactivate
```

---

## 🔄 Git Workflow / Git工作流程

### Initial Setup / 初始设置

```bash
# Initialize git repository (if not already done)
# 初始化git仓库（如果尚未完成）
git init

# Add remote repository
# 添加远程仓库
git remote add origin https://github.com/your-username/your-repo.git

# Create requirements.txt
# 创建依赖文件
pip freeze > requirements.txt
```

### Stage and Commit Changes / 暂存并提交更改

```bash
# Check current status / 检查当前状态
git status

# Stage files / 暂存文件
git add README.md requirements.txt
git add logistic_regression.py
git add *.ipynb
git add *.txt

# Or stage all files / 或暂存所有文件
git add .

# Commit changes / 提交更改
git commit -m "Add logistic regression implementation and documentation"
git commit -m "添加逻辑回归实现和文档"

# Push to GitHub / 推送到GitHub
git push origin main

# If pushing for the first time / 如果是第一次推送
git push -u origin main
```

### Update Existing Repository / 更新现有仓库

```bash
# 1. Stage the new files / 暂存新文件
git add README.md requirements.txt

# 2. Commit the changes / 提交更改
git commit -m "Add bilingual README and requirements"

# 3. Push to GitHub / 推送到GitHub
git push origin main
```

---

## 📈 Performance Analysis / 性能分析

The Jupyter notebook includes:
Jupyter Notebook 包含：

1. **Data Exploration** / 数据探索
   - Dataset statistics / 数据集统计
   - Feature distribution / 特征分布
   
2. **Training Analysis** / 训练分析
   - Loss convergence plot / 损失收敛图
   - Accuracy over iterations / 迭代过程中的准确率
   
3. **Model Evaluation** / 模型评估
   - Confusion matrix / 混淆矩阵
   - Precision & Recall / 精确率和召回率
   - F1 Score / F1分数
   
4. **Visualization** / 可视化
   - Decision boundary / 决策边界
   - Learning curve / 学习曲线

---

## 📝 Conclusion / 结论

This assignment demonstrates the fundamental mechanics of Logistic Regression and the importance of hyperparameter tuning (learning rate, iterations) in achieving a robust decision boundary.

本次作业展示了逻辑回归的基本机制以及超参数调整（学习率、迭代次数）在实现稳健决策边界中的重要性。

### Key Takeaways / 关键要点

| English | 中文 |
|---------|------|
| Logistic Regression is a linear model for binary classification | 逻辑回归是用于二分类的线性模型 |
| Sigmoid function maps outputs to probabilities between 0 and 1 | Sigmoid函数将输出映射到0到1之间的概率 |
| Binary Cross-Entropy is the appropriate loss function | 二分类交叉熵是合适的损失函数 |
| Gradient descent optimizes the weights iteratively | 梯度下降迭代优化权重 |
| Learning rate affects convergence speed and stability | 学习率影响收敛速度和稳定性 |

---

## 🔗 References / 参考资料

- [Logistic Regression - Wikipedia](https://en.wikipedia.org/wiki/Logistic_regression)
- [NumPy Documentation](https://numpy.org/doc/stable/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [Jupyter Notebook Documentation](https://jupyter-notebook.readthedocs.io/)

---

## 👤 Author / 作者

- **CHEN XI** (25053692)

---

## 📄 License / 许可证

This project is open source and available under the [MIT License](LICENSE).

本项目开源，基于 [MIT 许可证](LICENSE)。

✅ Dataset format explanation
✅ Expected results and performance metrics
✅ Professional GitHub formatting
