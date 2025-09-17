import itertools
import json
import re
import datasets
import glob
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from datasets import Dataset, DatasetDict

_SPLIT_DATA_PATH = '/data/gongoubo/data/WuDaoCorpus2.0_base_200G/*'
# 缓存文件
_CACHE_TRAIN_DATA_PATH = '/data/gongoubo/data/WuDaoCorpus2.0_base_200G_cache/'

feats = datasets.Features({"text": datasets.Value('string')})


def load_dataset(num_proc=1, **kargs):
    cache_dict_paths = glob.glob(os.path.join(_CACHE_TRAIN_DATA_PATH, '*'))
    ds = []
    res = []
    p = ProcessPoolExecutor(max_workers=num_proc)
    for path in cache_dict_paths:
        res.append(p.submit(datasets.load_from_disk,
                            path, **kargs))

    p.shutdown(wait=True)
    for future in tqdm(res, total=len(res)):
        ds.append(future.result())

    print(datasets.DatasetDict({"train": datasets.concatenate_datasets(ds)}))
    return datasets.DatasetDict({"train": datasets.concatenate_datasets(ds)})


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


def combine_title_content(example):
    # 拼接 title 和 content 列，用 "\n" 作为分隔符
    # example['text'] = example['title'] + "\n" + example['content']
    # return {'text': example['text']}  # 仅保留新生成的 'text'
    text = str(example['title']) + "\n" + str(example['content'])
    texts = split_sentence(text)
    return merge_sentences(texts)


def _generate_cache_arrow(index, path):
    print('saving dataset shard {}'.format(index))
    # ds = (datasets.load_dataset('json', data_files=path,
    #                             cache_dir='/cognitive_comp/gaoxinyu/data/huggingface-cache',
    #                             features=feats)['train'])
    ds = datasets.load_dataset('json', data_files=path,
                               cache_dir='/data/gongoubo/data/huggingface-cache',
                               )
    cols = (ds.column_names)["train"]
    ds = ds.map(combine_title_content, remove_columns=list(cols), features=feats)["train"]

    texts = list(itertools.chain.from_iterable([eval(i) for i in ds["text"]]))
    ds2 = (Dataset.from_dict({"text": texts}))

    ds2.save_to_disk(os.path.join(_CACHE_TRAIN_DATA_PATH, os.path.basename(path)))
    return 'saving dataset shard {} done'.format(index)


def generate_cache_arrow(num_proc=1) -> None:
    '''
    生成HF支持的缓存文件，加速后续的加载
    '''
    data_dict_paths = glob.glob(_SPLIT_DATA_PATH)
    p = ProcessPoolExecutor(max_workers=num_proc)
    res = []

    for index, path in enumerate(data_dict_paths):
        res.append(p.submit(_generate_cache_arrow, index, path))

    p.shutdown(wait=True)
    for future in res:
        print(future.result(), flush=True)


if __name__ == '__main__':
    # generate_cache_arrow(num_proc=64)
    load_dataset(num_proc=64)

