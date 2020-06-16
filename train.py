import os, sys, time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data
import torchvision.models as models

from batch import get_loader
from args import get_parser
from loss import *
from net import bninception

parser = get_parser()
opts = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu 

def norm(inputs, p=2, dim=-1, eps=1e-12):
	return inputs/inputs.norm(p,dim,keepdim=True).clamp(min=eps).expand_as(inputs)

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


class ImageModel(nn.Module):
	def __init__(self, emb_dim, pretrained=False):
		super(ImageModel,self).__init__()

		self.emb_dim = emb_dim
		self.pretrained = pretrained
		self.img_model = bninception(self.emb_dim)
		self.embedding = nn.Linear(1024, self.emb_dim)


	def forward(self, x):
		feat = self.img_model(x)
		emb = self.embedding(feat)
		emb = F.normalize(emb, p=2, dim=1)

		return feat, emb


def main():
	model = ImageModel(emb_dim=opts.embDim,pretrained=True)
	model.cuda()

	criterion = OurLoss(scale=opts.scale,
						M=opts.margin,
						dim=opts.embDim,
						cN=opts.class_split,
						kernel_muls=opts.kernelMuls,
						kernel_nums=opts.kernelNums,
						fix_sigmas=opts.sigmas,
						alpha=opts.alpha).cuda()

	optimizer = torch.optim.Adam([
					{'params': model.embedding.parameters()},
					{'params': model.img_model.parameters(), 'lr':opts.lr_img},
					{'params': criterion.parameters(), 'lr':opts.lr_cent}],
					lr = opts.lr, eps=opts.eps, weight_decay=opts.weight_decay)
	

	start_epoch = 0
	ckpts = []

	# preparing the training loader
	train_loader, test_loader = get_loader(opts)
	print('data loader prepared.')

	data_time = 0.0
	batch_time = 0.0
	test_time = 0.0
	start_time = time.time()
	end = time.time()

	for cur_epoch in range(start_epoch, opts.epochs):
		adjust_learning_rate(optimizer, cur_epoch, opts)
		
		data_t, batch_t = train_epoch(model, criterion, train_loader, optimizer, cur_epoch, opts.freeze_BN)
		data_time += data_t
		batch_time += batch_t

		if (cur_epoch+1) % opts.valfreq == 0:
			end = time.time()
			test_acc = test(model, test_loader)
			test_time += time.time() - end
		print('data_time:', data_time, 'batch_time:', batch_time, 'test_time:', test_time)

	# test
	test(model, test_loader)
	print('total time:', time.time()-start_time)


def train_epoch(model, criterion, train_loader, optimizer, epoch, freeze_BN):
	print('epoch:', epoch)

	data_time = 0.0
	batch_time = 0.0
	end = time.time()

	model.train()
	if freeze_BN:
		for m in model.img_model.modules():
			if isinstance(m, nn.BatchNorm2d):
				m.eval()

	for i, (inputs, labels, index) in enumerate(train_loader):
		data_time += time.time() - end
		end = time.time()

		input_var = torch.autograd.Variable(inputs).cuda()
		label_var = torch.autograd.Variable(labels).cuda()

		optimizer.zero_grad()
		pooled, emb = model(input_var)
		loss = criterion([pooled], emb, label_var)
		loss.backward()

		if opts.gradclip > 0:
			params = []
			for group in optimizer.param_groups:
				for p in group['params']:
					params.append(p)
			torch.nn.utils.clip_grad_norm_(params, opts.gradclip)

		optimizer.step()

		if (i % 100) == 0:
			print(loss.item(), 'epoch', epoch, 'batch', i, 'finish')

		batch_time += time.time() - end
		end = time.time()

	return data_time, batch_time

def test(model, test_loader):
	print('start testing...')
	model.eval()
	with torch.no_grad():
		for i, (inputs, labels, index) in enumerate(test_loader):
			input_var = torch.autograd.Variable(inputs).cuda()
			label_var = torch.autograd.Variable(labels).cuda()
			pooled, emb = model(input_var)
			if i==0:
				emb_np = emb.data.cpu().numpy()
				class_np = label_var.data.cpu().numpy()
			else:
				emb_np = np.concatenate((emb_np, emb.data.cpu().numpy()),axis=0)
				class_np = np.concatenate((class_np, label_var.data.cpu().numpy()),axis=0)
	
	recall = evaluation(opts, emb_np, emb_np, class_np)
	print('recall:', recall)
	return recall[0]

def euclidean_dist(x, y):
	bx = x.shape[0]
	by = y.shape[0]
	
	xx = np.sum(np.power(x, 2), 1, keepdims=True).repeat(by, 1)
	yy = np.sum(np.power(y, 2), 1, keepdims=True).repeat(bx, 1).T
	dist = xx + yy - 2*np.dot(x, y.T)
	return -1.0*dist

def cos_sim(x, y):
	return x.dot(y.T)

def evaluation(opts, source, target, classes):
	num = source.shape[0]
	Kset = [1,2,4,8]
	kmax = np.max(Kset)
	recallK = np.zeros(len(Kset))

	#compute Recall@K
	sim = cos_sim(source, target)
	minval = np.min(sim) - 1.
	sim -= np.diag(np.diag(sim))
	sim += np.diag(np.ones(num) * minval)
	indices = np.argsort(-sim, axis=1)[:, : kmax]
	YNN = classes[indices]
	for i in range(0, len(Kset)):
		pos = 0.
		for j in range(0, num):
			if classes[j] in YNN[j, :Kset[i]]:
				pos += 1.
		recallK[i] = pos/num
	return recallK

def adjust_learning_rate(optimizer, epoch, opts):
	if (epoch+1)%opts.lr_update == 0:
		for param_group in optimizer.param_groups:
			param_group['lr'] = param_group['lr']*opts.decay_rate

if __name__ == '__main__':
	setup_seed(opts.seed)
	opts.class_file = os.path.join(opts.data_path, opts.class_file)
	opts.imgs_path_file = os.path.join(opts.data_path, opts.imgs_path_file)
	opts.image_dir = os.path.join(opts.data_path, opts.image_dir)
	print(opts)
	main()

