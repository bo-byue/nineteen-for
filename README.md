 题目四 4B 代码部分文件说明

一、各个主要代码分别有什么作用？

 1. `01_check_environment.py`

这个文件用于检查实验环境是否配置成功。

它主要输出：

Python 版本
PyTorch 版本
Torchvision 版本
是否支持 CUDA

这个文件的作用是证明代码运行环境已经准备好，比如我们使用的是 Python 3.12.4，PyTorch 和 Torchvision 也已经安装成功。

PPT 中可以把它放在“实验环境”部分。

---

 2. `02_show_omniglot_sample.py`

这个文件用于展示 Omniglot 数据集中的单张样例图片。

它会从数据集中读取第一张图片，输出图片形状和标签，并保存一张单字符样例图到 `outputs` 文件夹。

这个文件的作用是帮助观众直观看到：我们使用的数据不是普通数字图片，而是 Omniglot 中的手写字符图片。

PPT 中可以把它放在“数据集介绍”部分。

---

 3. `03_visualize_5way5shot_episode.py`

这个文件用于构造并展示一次 5-way 5-shot 小样本分类任务。

5-way 表示一次任务随机选择 5 个类别。
5-shot 表示每个类别提供 5 张支持集图片。
同时，每个类别还会抽取 15 张查询集图片用于分类测试。

所以一次 episode 中：

Support Set 支持集：5 × 5 = 25 张图片
Query Set 查询集：5 × 15 = 75 张图片

这个文件会输出 support 和 query 的张量维度，并保存两张图：

5-way 5-shot 支持集采样结果
查询集样例展示

这个文件的作用是证明我们的任务采样逻辑是正确的，也就是确实构造出了 5-way 5-shot 小样本任务。

PPT 中可以把它放在“任务构造”或“数据采样过程”部分。

---

 4. `04_train_protonet_5way5shot.py`

这个文件是最核心的正式训练代码。

它实现了完整的 Prototypical Network 原型网络流程，包括：

加载 Omniglot 训练集和测试集
随机构造 5-way 5-shot episode
使用 CNN 提取图像特征
根据 support set 计算每个类别的原型
计算 query 图片与各类别原型之间的距离
根据距离完成分类
训练模型并测试最终准确率
保存 loss 曲线、accuracy 曲线和最终结果

这个文件运行后，会输出训练过程，例如：

Episode 50/300
Episode 100/300
Episode 150/300
Episode 200/300
Episode 250/300
Episode 300/300

最后会输出测试结果，例如：

5-way 5-shot 测试结果
平均准确率：97.77%

这个文件是我们 4B 代码部分最重要的成果，可以放在 PPT 的“模型方法”“训练过程”和“实验结果”部分。

---

 二、outputs 文件夹里的九个结果分别是什么

1. `01_environment_check.png`

中文名：实验环境检查结果

这张图是 PowerShell 中运行 `01_check_environment.py` 的截图。

它用于证明 Python、PyTorch、Torchvision 等环境已经安装成功，代码可以正常运行。



---

 2. `02_omniglot_single_sample.png`

中文名：Omniglot 数据集单张样例

这张图展示了 Omniglot 数据集中的一个手写字符图片。

它的作用是让观众直观看到我们处理的图像数据是什么样的。


---

 3. `03_episode_support_set_5way5shot.png`

中文名：5-way 5-shot 支持集采样结果

这张图展示了一次小样本任务中的 support set。

图中共有 25 张图片，对应：

5 个类别
每个类别 5 张图片

它用于说明 5-way 5-shot 的任务构造方式。



---

 4. `04_episode_query_set_samples.png`

中文名：查询集样例展示

这张图展示了一次 episode 中 query set 的部分样例图片。

Query set 的作用是让模型在看过 support set 后，对这些新图片进行分类。


---

 5. `05_powershell_episode_shape_output.png`

中文名：Episode 数据维度验证结果

这张图是 PowerShell 输出截图，主要展示：

support_images 形状：torch.Size([25, 1, 28, 28])
support_labels 形状：torch.Size([25])
query_images 形状：torch.Size([75, 1, 28, 28])
query_labels 形状：torch.Size([75])

它用于证明我们的采样数量是正确的：

Support Set：25 张
Query Set：75 张

其中 `[1, 28, 28]` 表示每张图片是单通道灰度图，大小为 28×28。



---

 6. `06_training_loss_curve.png`

中文名：训练损失变化曲线

这张图展示训练过程中 loss 的变化。

Loss 表示模型预测结果和真实标签之间的差距。通常来说，loss 越低，说明模型学习效果越好。

这张图可以用来说明模型在训练过程中逐渐收敛。



---

 7. `07_training_accuracy_curve.png`

中文名：训练准确率变化曲线

这张图展示训练过程中 accuracy 的变化。

Accuracy 表示模型在 query set 上分类正确的比例。通常来说，accuracy 越高，说明分类效果越好。

这张图可以和 loss 曲线一起说明训练过程是有效的。



---

 8. `08_final_accuracy_result.txt`

中文名：最终测试准确率文本结果

这个不是图片，而是程序自动保存的最终测试结果文本。

里面记录了最终测试准确率，例如：

5-way 5-shot 测试结果
平均准确率：97.77%

它可以作为实验结果的文字备份，PPT 中不一定要展示，但可以用于报告或答辩时核对结果。

---

 9. `09_powershell_training_log.png`

中文名：模型训练过程输出

这张图是 PowerShell 中训练过程的截图。

它展示了模型从 Episode 50 到 Episode 300 的训练日志，包括每一阶段的 loss 和 accuracy。

它的作用是证明模型确实进行了训练，而不是只展示最终结果。



三、环境配置
1. 创建并激活 Python 虚拟环境（推荐 Python ≥ 3.8）：
   ```bash
   python -m venv venv
   source venv/bin/activate    Linux / macOS
   venv\Scripts\activate       Windows
安装依赖：
Bash
pip install -r requirements.txt
若使用 CUDA，请根据 PyTorch 官方指南 安装对应版本。




四、运行方法
按顺序执行以下脚本：

Bash
 1. 检查环境
python 01_check_environment.py
 2. 查看 Omniglot 样例图片
python 02_show_omniglot_sample.py
 3. 可视化一个 episode 的 support/query 划分
python 03_visualize_5way5shot_episode.py
 4. 训练原型网络并进行 5‑way 5‑shot 测试
python 04_train_protonet_5way5shot.py
所有输出图片和训练曲线将保存在 outputs/ 目录下。

实验结果
在默认超参数下（300 个训练 episodes，5‑way 5‑shot），模型在 Omniglot 测试集上的平均准确率约为 98% 左右。

最终准确率文本将保存至 outputs/08_final_accuracy_result.txt。

注意事项
首次运行脚本时会自动下载 Omniglot 数据集（约 18 MB）至 ./data 目录，请保持网络畅通。
训练过程默认使用 CPU（若无 NVIDIA GPU），可通过修改脚本中的 device 变量切换设备。

