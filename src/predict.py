import torch
import config
from model import InputMethodModel

from tokenizer import JiebaTokenizer


def predict_batch(model, inputs):
	"""
	批量预测
	:param model: 模型
	:param inputs: 输入 形状：(batch_size, seq_len)
	:return: 预测结果 形状： (batch_size, 5)
	"""
	model.eval()
	with torch.no_grad():
		output = model(inputs)
	# output.shape (batch_size, vocab_size)
	# 获取排名前五的index
	top5_indexes = torch.topk(output, k=5).indices
	# top5_indexes.shape (batch_size, 5)
	top5_indexes_list = top5_indexes.tolist()
	return top5_indexes_list


def predict(text, tokenizer, model, device):
	# 1. 处理输入
	indexes = tokenizer.encode(text)
	# input.shape (batch_size, seq_len)
	input_tensor = torch.tensor([indexes], dtype=torch.long).to(device)
	# 2. 预测逻辑
	top5_indexes_list = predict_batch(model, input_tensor)
	top5_tokens = [tokenizer.index2word[index] for index in top5_indexes_list]
	return top5_tokens


def run_predict():
	# 1. 确定设备
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	# 2. 加载词表
	tokenizer = JiebaTokenizer.from_vocab(config.MODEL_DIR / 'vocab.txt')
	# 3. 加载模型
	model = InputMethodModel(vocab_size=tokenizer.vocab_size).to(device)
	model.load_state_dict(torch.load(config.MODEL_DIR / 'best.pt'))
	print("欢迎使用输入法模型（输入q或者quit退出）")
	input_history = ''
	while True:
		user_input = input(">")
		if user_input in ['q', 'quit']:
			print("bye！")
			break
		if user_input.strip() == '':
			print("请输入内容")
			continue
		input_history += user_input
		print(f"历史数据：{input_history}")
		top5_tokens = predict(input_history, tokenizer, model, device)
		print(f"预测结果：{top5_tokens}")


if __name__ == '__main__':
	run_predict()
