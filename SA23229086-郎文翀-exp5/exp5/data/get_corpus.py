import json

if __name__ == "__main__":
    # 定义文件列表和输出路径
    files = ['train', 'val', 'test']
    ch_path = 'corpus.ch'
    en_path = 'corpus.en'
    ch_lines = []  # 存储中文语料的列表
    en_lines = []  # 存储英文语料的列表

    # 遍历文件列表
    for file in files:
        # 但问题是，汉语、日语等语言的字与字之间并没有空格分隔。sentencepiece提出，可以将所有字符编码成Unicode码（包括空格），通过训练直接将原始文本（未切分）变为分词后的文本，从而避免了跨语言的问题。
        # 从 JSON 文件加载语料
        corpus = json.load(open('./data/json/' + file + '.json', 'r'))
        # 遍历每个语料项，将中文和英文分别添加到对应的列表中
        for item in corpus:
            ch_lines.append(item[1] + '\n')
            en_lines.append(item[0] + '\n')

    # 将中文语料写入文件（使用 UTF-8 编码）
    with open(ch_path, "w", encoding="utf-8") as fch:
        fch.writelines(ch_lines)

    # 将英文语料写入文件（使用 UTF-8 编码）
    with open(en_path, "w", encoding="utf-8") as fen:
        fen.writelines(en_lines)

    # 输出中文语料的行数
    print("中文语料行数: ", len(ch_lines))
    # 输出英文语料的行数
    print("英文语料行数: ", len(en_lines))
    print("-------- 获取语料！--------")
