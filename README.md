<h1 id="AdxoK">融合方面级情感分析的模型说明</h1>
<h2 id="fjRFR">项目简介</h2>
欢迎来到我们的项目——融合方面级情感分析的模型。本项目耦合在我们的大创项目中。核心模型基于哈工大讯飞实验室的RoBERTa-wwm-ext，实现了对旅游评论的方面级情感分析，为用户提供更加精准的旅游建议。

<h2 id="fd1Rq">模型介绍</h2>
<h3 id="PouPf">1. 基础模型：RoBERTa-wwm-ext</h3>
本项目采用的模型是基于哈工大讯飞实验室的RoBERTa-wwm-ext。该模型通过全词掩码（Whole Word Masking）技术对中文BERT进行预训练，有效提升了模型在中文自然语言处理任务中的性能。

<h3 id="fRbgL">2. 方面级情感分析</h3>
方面级情感分析（Aspect-Based Sentiment Analysis, ABSA）是自然语言处理领域的一个重要任务。本项目的模型能够识别旅游评论中的不同方面（如景点、住宿、交通等），并针对这些方面进行情感分析，从而为用户提供更具体的旅游建议。

<h2 id="J81Q6">项目资源</h2>
+ Gitee：[natural-language-processing/Chinese-BERT-wwm](https://gitee.com/natural-language-processing/Chinese-BERT-wwm)
+ HuggingFace Hub：[https://huggingface.co/hfl/chinese-roberta-wwm-ext](https://huggingface.co/hfl/chinese-roberta-wwm-ext)
+ Paper：[Pre-Training with Whole Word Masking for Chinese BERT](https://arxiv.org/abs/1906.08101)
+ Data：[https://github.com/Meituan-Dianping/asap](https://github.com/Meituan-Dianping/asap)

<h2 id="WQvsm">快速开始</h2>
<h3 id="JvTww">环境配置(本地环境/服务器)</h3>
1. Python版本：3.10+
2. 依赖库：PyTorch、Transformer等
3. RoBERTa权重文件以及第一代模型文件（本地环境使用）[https://drive.google.com/drive/folders/10zGEPVntXXa-YV2RFkbygrCdWwc2PXvP?usp=drive_link](https://drive.google.com/drive/folders/10zGEPVntXXa-YV2RFkbygrCdWwc2PXvP?usp=drive_link)
4. 服务器环境：[https://huggingface.co/hfl/chinese-roberta-wwm-ext](https://huggingface.co/hfl/chinese-roberta-wwm-ext)

```python
# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("fill-mask", model="hfl/chinese-roberta-wwm-ext")
```

```python
# Load model directly
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
model = AutoModelForMaskedLM.from_pretrained("hfl/chinese-roberta-wwm-ext")
```

<h3 id="NLwwd">模型训练</h3>

```plain
Moudle/
│
├── generation1/
│   ├── train.py
│   └── utils.py
│   
├── generation2/
│   └── TensorBoard_train.py
│   └── utils.py
│
└── README.md
```

修改文件中的模型路径即可开始训练



<h2 id="AqfvW">贡献者</h2>
感谢以下贡献者为本项目付出的努力：

[Horizon-Leo - Overview](https://github.com/Horizon-Leo)

<h2 id="Q9g0O">许可证</h2>
本项目遵循MIT许可证，详细内容请参考LICENSE文件。

<h2 id="A9KsZ">交流与反馈</h2>
如果您在使用过程中遇到问题或有任何建议，请通过以下方式与我们联系：

+ 提交Issue：
+ 加入QQ群：（验证信息：）  
再次感谢您对本项目的关注与支持，希望我们的努力能为您的旅游体验带来更多便捷与愉悦！


















