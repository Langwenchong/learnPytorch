from torch.autograd import Variable
from utils import english_tokenizer_load
from utils import chinese_tokenizer_load

import config

DEVICE = config.device


def subsequent_mask(size):
    """Mask out subsequent positions."""
    # 设定subsequent_mask矩阵的shape
    attn_shape = (1, size, size)

    # 生成一个右上角(不含主对角线)为全1，左下角(含主对角线)为全0的subsequent_mask矩阵
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')

    # 返回一个右上角(不含主对角线)为全False，左下角(含主对角线)为全True的subsequent_mask矩阵
    return torch.from_numpy(subsequent_mask) == 0


class Batch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, src_text, trg_text, src, trg=None, pad=0):
        self.src_text = src_text
        self.trg_text = trg_text
        src = src.to(DEVICE)
        self.src = src
        # 对于当前输入的句子非空部分进行判断成bool序列
        # 并在seq length前面增加一维，形成维度为 1×seq length 的矩阵
        self.src_mask = (src != pad).unsqueeze(-2)
        # 如果输出目标不为空，则需要对decoder要使用到的target句子进行mask
        if trg is not None:
            trg = trg.to(DEVICE)
            # decoder要用到的target输入部分
            self.trg = trg[:, :-1]
            # decoder训练时应预测输出的target结果
            self.trg_y = trg[:, 1:]
            # 将target输入部分进行attention mask
            self.trg_mask = self.make_std_mask(self.trg, pad)
            # 将应输出的target结果中实际的词数进行统计
            self.ntokens = (self.trg_y != pad).data.sum()

    # Mask掩码操作
    @staticmethod
    def make_std_mask(tgt, pad):
        """Create a mask to hide padding and future words."""
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


import json
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class MTDataset(Dataset):
    def __init__(self, data_path):
        # 初始化函数，读取数据集，加载分词器，并设置PAD、BOS、EOS标记
        self.out_en_sent, self.out_cn_sent = self.get_dataset(data_path, sort=True)
        self.sp_eng = english_tokenizer_load()  # 英文分词器
        self.sp_chn = chinese_tokenizer_load()  # 中文分词器
        self.PAD = self.sp_eng.pad_id()  # PAD标记的ID（用于填充）
        self.BOS = self.sp_eng.bos_id()  # BOS标记的ID（句子开头）
        self.EOS = self.sp_eng.eos_id()  # EOS标记的ID（句子结尾）

    @staticmethod
    def len_argsort(seq):
        """传入一系列句子数据(分好词的列表形式)，按照句子长度排序后，返回排序后原来各句子在数据中的索引下标"""
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    def get_dataset(self, data_path, sort=False):
        """把中文和英文按照同样的顺序排序, 以英文句子长度排序的(句子下标)顺序为基准"""
        dataset = json.load(open(data_path, 'r'))  # 从JSON文件中加载数据集
        out_en_sent = []
        out_cn_sent = []
        for idx, _ in enumerate(dataset):
            out_en_sent.append(dataset[idx][0])  # 英文句子
            out_cn_sent.append(dataset[idx][1])  # 中文句子
        if sort:
            sorted_index = self.len_argsort(out_en_sent)  # 根据英文句子长度排序
            out_en_sent = [out_en_sent[i] for i in sorted_index]
            out_cn_sent = [out_cn_sent[i] for i in sorted_index]
        return out_en_sent, out_cn_sent

    def __getitem__(self, idx):
        # 获取指定索引的样本，返回英文句子和中文句子
        eng_text = self.out_en_sent[idx]
        chn_text = self.out_cn_sent[idx]
        return [eng_text, chn_text]

    def __len__(self):
        # 返回数据集的长度
        return len(self.out_en_sent)

    def collate_fn(self, batch):
        # 用于在 DataLoader 中处理批量数据的函数
        src_text = [x[0] for x in batch]  # 英文句子
        tgt_text = [x[1] for x in batch]  # 中文句子

        # 将英文句子和中文句子转换为ID序列，并在每个句子的开头和结尾添加BOS和EOS标记
        src_tokens = [[self.BOS] + self.sp_eng.EncodeAsIds(sent) + [self.EOS] for sent in src_text]
        tgt_tokens = [[self.BOS] + self.sp_chn.EncodeAsIds(sent) + [self.EOS] for sent in tgt_text]

        # 使用pad_sequence进行填充，将每个batch的句子填充到相同的长度
        batch_input = pad_sequence([torch.LongTensor(np.array(l_)) for l_ in src_tokens],
                                   batch_first=True, padding_value=self.PAD)
        batch_target = pad_sequence([torch.LongTensor(np.array(l_)) for l_ in tgt_tokens],
                                    batch_first=True, padding_value=self.PAD)

        # 返回一个Batch对象，包含原始文本、ID序列和填充后的ID序列
        return Batch(src_text, tgt_text, batch_input, batch_target, self.PAD)
