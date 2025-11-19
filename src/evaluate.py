import torch
import config
from model import InputMethodModel
from dataset import get_dataloader
from predict import predict_batch

from tokenizer import JiebaTokenizer


def evaluate(model, test_loader, device):
	top1_acc_count = 0
	top5_acc_count = 0
	total_count = 0
	for inputs, targets in test_loader:
		inputs = inputs.to(device)
		# input.shape: (batch_size, seq_len)
		targets = targets.tolist()
		# target.shape (batch_size)
		# 获取预测前五的token
		top5_indexes_list = predict_batch(model, inputs)
		# 计算准确度
		for target, top5_indexes in zip(targets, top5_indexes_list):
			total_count += 1
			if target == top5_indexes[0]:
				top1_acc_count += 1
			if target in top5_indexes:
				top5_acc_count += 1
		return top1_acc_count / total_count, top5_acc_count / total_count


def run_evaluate():
	# 1. 确定设备
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	# 2. 加载词表
	tokenizer = JiebaTokenizer.from_vocab(config.MODEL_DIR / 'vocab.txt')
	# 3. 加载模型
	model = InputMethodModel(vocab_size=tokenizer.vocab_size).to(device)
	model.load_state_dict(torch.load(config.MODEL_DIR / 'best.pt'))
	# 4. 数据集
	test_loader = get_dataloader(train=False)
	# 5. 评估
	top1_acc, top5_acc = evaluate(model, test_loader, device)
	print(f"top1_acc: {top1_acc}, top5_acc: {top5_acc}")


if __name__ == '__main__':
	run_evaluate()
