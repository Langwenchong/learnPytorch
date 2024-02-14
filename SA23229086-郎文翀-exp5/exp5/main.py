import utils
import config
import logging
import numpy as np

import torch
from torch.utils.data import DataLoader

from train import train, test, translate
from data_loader import MTDataset
from utils import english_tokenizer_load
from model import make_model, LabelSmoothing


class NoamOpt:
    """
    实现Noam学习率调度的优化器包装器。
    """

    def __init__(self, model_size, factor, warmup, optimizer):
        """
        初始化NoamOpt优化器。

        参数：
        - model_size：模型的大小。
        - factor：学习率的缩放因子。
        - warmup：预热步数。
        - optimizer：基础优化器。

        属性：
        - optimizer：基础优化器。
        - _step：用于跟踪更新的内部步数计数器。
        - warmup：预热步数。
        - factor：学习率的缩放因子。
        - model_size：模型的大小。
        - _rate：当前学习率。
        """
        self.optimizer = optimizer
        self._step = 0
        self.factor = factor
        self.model_size = model_size
        self.warmup = warmup

    def step(self):
        """
        更新参数和学习率。
        """
        self._step += 1
        rate = self.rate()
        # 更新优化器中的学习率
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        """
        实现上面的 `lrate` 函数。
        """
        if step is None:
            step = self._step
        # 实现学习率调度
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
    """
    获取标准优化器，使用Noam学习率调度。

    参数：
    - model：要优化的模型。

    返回：
    - NoamOpt：使用Noam学习率调度的优化器。
    """
    # 对于 batch_size 为 32，一个 epoch 需要 5530 步，预热 2 个 epoch
    return NoamOpt(model.src_embed[0].d_model, 1, 10000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


def run():
    """
    主训练函数。

    该函数执行模型训练的主要逻辑，包括数据集加载、模型初始化、损失函数设置、优化器设置、训练和测试。

    注意：函数中使用的配置参数（例如 config.src_vocab_size、config.tgt_vocab_size 等）需要在 config 模块中定义。

    返回：无
    """
    # 设置日志
    utils.set_logger(config.log_path)

    # 加载训练、开发和测试数据集
    train_dataset = MTDataset(config.train_data_path)
    dev_dataset = MTDataset(config.dev_data_path)
    test_dataset = MTDataset(config.test_data_path)

    logging.info("-------- 数据集构建完成！--------")

    # 创建数据加载器
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size,
                                  collate_fn=train_dataset.collate_fn)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=config.batch_size,
                                collate_fn=dev_dataset.collate_fn)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=config.batch_size,
                                 collate_fn=test_dataset.collate_fn)

    logging.info("-------- 获取数据加载器！--------")

    # 初始化模型
    model = make_model(config.src_vocab_size, config.tgt_vocab_size, config.n_layers,
                       config.d_model, config.d_ff, config.n_heads, config.dropout)
    # 使用 DataParallel 将模型放到多个 GPU 上进行训练
    model_par = torch.nn.DataParallel(model)

    # 设置损失函数
    if config.use_smoothing:
        # 使用 LabelSmoothing 损失函数
        criterion = LabelSmoothing(
            size=config.tgt_vocab_size, padding_idx=config.padding_idx, smoothing=0.1)
        criterion.cuda()
    else:
        # 使用交叉熵损失函数
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='sum')

    # 设置优化器
    if config.use_noamopt:
        # 使用 Noam 优化器
        optimizer = get_std_opt(model)
    else:
        # 使用 AdamW 优化器
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    # 进行训练和测试
    train(train_dataloader, dev_dataloader,
          model, model_par, criterion, optimizer)
    test(test_dataloader, model, criterion)


def check_opt():
    """
    检查学习率变化的函数。

    该函数通过绘制学习率在不同超参数设置下随时间的变化图表，以帮助调试和优化模型的学习率调度。

    注意：函数中使用的配置参数（例如 config.src_vocab_size、config.tgt_vocab_size 等）需要在 config 模块中定义。

    返回：无
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # 创建模型和优化器
    model = make_model(config.src_vocab_size, config.tgt_vocab_size, config.n_layers,
                       config.d_model, config.d_ff, config.n_heads, config.dropout)
    opt = get_std_opt(model)

    # 三种学习率超参数设置
    opts = [opt,
            NoamOpt(512, 1, 20000, None),
            NoamOpt(256, 1, 10000, None)]

    # 绘制学习率随时间变化的图表
    # plt.plot(np.arange(1, 50000), [[opt.rate(i) for opt in opts] for i in range(1, 50000)])
    # plt.legend(["512:10000", "512:20000", "256:10000"])
    # plt.show()


def one_sentence_translate(sent, beam_search=True):
    """
    对单个句子进行翻译的函数。

    参数：
    - sent：待翻译的输入句子。
    - beam_search：是否使用束搜索，默认为 True。

    注意：函数中使用的配置参数（例如 config.src_vocab_size、config.tgt_vocab_size 等）需要在 config 模块中定义。

    返回：无
    """
    # 初始化模型
    model = make_model(config.src_vocab_size, config.tgt_vocab_size, config.n_layers,
                       config.d_model, config.d_ff, config.n_heads, config.dropout)

    # 获取开始和结束标记
    BOS = english_tokenizer_load().bos_id()  # 2
    EOS = english_tokenizer_load().eos_id()  # 3

    # 将输入句子进行编码
    src_tokens = [[BOS] + english_tokenizer_load().EncodeAsIds(sent) + [EOS]]
    batch_input = torch.LongTensor(np.array(src_tokens)).to(config.device)

    # 调用翻译函数
    translate(batch_input, model, use_beam=beam_search)


def translate_example():
    """
    单句翻译示例的函数。

    该函数提供了一个输入句子的示例，并使用 one_sentence_translate 函数进行翻译。

    注意：函数中使用的配置参数（例如 config.src_vocab_size、config.tgt_vocab_size 等）需要在 config 模块中定义。

    返回：无
    """
    # 示例句子
    sent = "While this was a worthy goal, historians will point out, it was far from the only imperative."

    # 调用单句翻译函数
    one_sentence_translate(sent, beam_search=True)


if __name__ == "__main__":
    import os

    # 设置可见的 CUDA 设备
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    import warnings

    # 忽略警告
    warnings.filterwarnings('ignore')

    # 运行单句翻译示例
    translate_example()

    # run()
