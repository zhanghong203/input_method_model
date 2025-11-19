import pandas as pd

from sklearn.model_selection import train_test_split
from tqdm import tqdm

import config
from tokenizer import JiebaTokenizer


def build_dataset(setences, tokenizer):
	indexed_setences = [tokenizer.encode(setence) for setence in setences]
	# 构建训练集
	dataset = []
	for setence in tqdm(indexed_setences, desc="构建训练集"):
		for i in range(len(setence) - config.SQL_LEN):
			input = setence[i:i + config.SQL_LEN]
			output = setence[i + config.SQL_LEN]
			dataset.append({'input': input, 'output': output})
	return dataset


def load_data():
	print("开始处理数据")
	# 加载语料
	df = pd.read_json(config.RAW_DATA_DIR / 'synthesized_.jsonl', orient='records', lines=True).sample(frac=0.01)
	# 提取句子
	sentences = []
	for dialog in df['dialog']:
		for sentence in dialog:
			sentences.append(sentence.split('：')[1])
	# print(len(sentences))
	train_setences, test_setences = train_test_split(sentences, test_size=0.2, random_state=42)
	# 构建词表
	JiebaTokenizer.build_vocab(train_setences, config.MODEL_DIR / 'vocab.txt')
	# 创建分词器对象
	tokenizer = JiebaTokenizer.from_vocab(config.MODEL_DIR / 'vocab.txt')
	# 保存训练集
	train_dataset = build_dataset(train_setences, tokenizer)
	pd.DataFrame(train_dataset).to_json(config.PROCESSED_DATA_DIR / 'train.jsonl', orient='records', lines=True)
	# 保存测试集
	test_dataset = build_dataset(test_setences, tokenizer)
	pd.DataFrame(test_dataset).to_json(config.PROCESSED_DATA_DIR / 'test.jsonl', orient='records', lines=True)
	print("结束处理数据")


if __name__ == '__main__':
	load_data()
