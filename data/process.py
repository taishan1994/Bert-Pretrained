# 将文本转换为{"text":""}的格式
import itertools
import re
import json
import random
import datasets
from glob import glob
from tqdm import tqdm


def process_wudao_200g():
    path = "/data/gongoubo/data/WuDaoCorpus2.0_base_200G/"
    files = glob(path + "*.json")
    print(len(files))
    # 读取出来并合并成一个
    total = 0
    for file in tqdm(files, total=len(files)):
        with open(file, "r") as fp:
            data = json.loads(fp.read())
        total += len(data)
        data = [d["title"] + "\b" + d["content"] for d in data]
        random.shuffle(data)

    print(total)


def split_sentence(text):
    """
    the first rank of sentence cut
    """
    sent = re.sub('([。！？\?])([^”’])', r"\1\n\2", text)  # 单字符断句符
    sent = re.sub('(\.{6})([^”’])', r"\1\n\2", sent)  # 英文省略号
    sent = re.sub('(\…{2})([^”’])', r"\1\n\2", sent)  # 中文省略号
    sent = re.sub('([。！？\?][”’])([^，。！？\?])', r"\1\n\2", sent)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后
    return sent.split("\n")


def merge_sentences(sentences, max_length=512):
    res = []
    merged_texts = []

    for i, sentence in enumerate(sentences):
        if len("".join(merged_texts)) + len(sentence) < max_length - 3:
            merged_texts.append(sentence)
        else:
            res.append("".join(merged_texts))
            merged_texts = [sentence]

    # 添加最后的文本（如果存在）
    if "".join(merged_texts) not in res:
        res.append("".join(merged_texts))

    return {"text": json.dumps(res, ensure_ascii=False)}


# 定义拼接函数
def combine_title_content(example):
    # 拼接 title 和 content 列，用 "\n" 作为分隔符
    text = example['title'] + "\n" + example['content']
    texts = split_sentence(text)
    # text = text.split("。")
    return merge_sentences(texts)
    # return {"text": text}
    # return [{"text": t} for t in text]


def test_load_one():
    path = "/data/gongoubo/data/WuDaoCorpus2.0_base_200G/part-2021278643.json"
    ds = datasets.load_dataset('json', data_files=path)
    print(ds)
    cols = (ds.column_names)["train"]
    print(cols)
    ds = ds.map(combine_title_content, remove_columns=list(cols))
    print(ds)
    text = list(itertools.chain.from_iterable([eval(i) for i in ds["train"]["text"]]))
    from datasets import Dataset, DatasetDict
    ds = Dataset.from_dict({"text": text})
    ds = DatasetDict({"train": ds})
    for i in ds["train"]["text"][:10]:
        print(len(i), i)


if __name__ == '__main__':
    # process_wudao_200g()
    test_load_one()