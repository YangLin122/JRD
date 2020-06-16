import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import BatchSampler

import os,sys
import numpy as np
import pickle as pkl
from PIL import Image
import scipy
from scipy import io
import pandas as pd

from args import get_parser

#parser = get_parser()
#opts = parser.parse_args()

def default_loader(path):
	try:
		img = Image.open(path).convert('RGB')
		return img
	except:
		print(path)
		return Image.new('RGB', (224,224), 'white')


class MyDataset_CUB(data.Dataset):
	def __init__(self, transform=None, partition=None, loader=default_loader, class_file='', imgs_path_file='', image_dir='', class_split=100):
		self.partition = partition
		if self.partition not in ['train', 'test', 'all']:
			raise Exception('unknown partition type %s.'%self.partition)

		self.class_split = class_split
		with open(class_file, 'r') as f:
			lines = f.readlines()
			self.ids = []
			self.classes = {}
			for line in lines:
				ind, label = line.strip().split()
				ind = int(ind)
				label = int(label)
				self.classes[ind] = label
				if self.partition == 'train':
					if label <= self.class_split:
						self.ids.append(ind)
				elif self.partition == 'test':
					if label > self.class_split:
						self.ids.append(ind)
				elif self.partition == 'all':
					self.ids.append(ind)

		self.class_indices = {}
		for index in range(len(self.ids)):
			class_id = self.classes[self.ids[index]]
			if class_id not in self.class_indices:
				self.class_indices[class_id] = []
			self.class_indices[class_id].append(index)


		with open(imgs_path_file, 'r') as f:
			lines = f.readlines()
			self.imgs_path = {}
			for line in lines:
				ind, path = line.strip().split()
				self.imgs_path[int(ind)] = path	

		self.transform = transform
		self.loader = loader
		self.image_dir = image_dir

	def __getitem__(self, index):
		ind = self.ids[index]

		path = os.path.join(self.image_dir, self.imgs_path[ind])
		label = self.classes[ind]

		img_feat = self.loader(path)
		if self.transform is not None:
			img_feat = self.transform(img_feat)

		return img_feat, label-1, index

	def __len__(self):
		return len(self.ids)

	def bootstrap(self, index):
		class_id = self.classes[self.ids[index]]
		self.class_indices[class_id].append(index)


def RGB2BGR(im):
	assert im.mode == 'RGB'
	r, g, b = im.split()
	return Image.merge('RGB', (b, g, r))


def get_loader(opts):
	dataset_class = None
	dataset_class = MyDataset_CUB

	normalize = transforms.Normalize(mean=[104., 117., 128.],
									std=[1., 1., 1.])


	train_trans = transforms.Compose([
					transforms.Lambda(RGB2BGR),
					transforms.RandomResizedCrop(224),
					transforms.RandomHorizontalFlip(),
					transforms.ToTensor(),
					transforms.Lambda(lambda x: x.mul(255)),
					normalize,
				])

	test_trans = transforms.Compose([
				transforms.Lambda(RGB2BGR),
				transforms.Resize(256), 
				transforms.CenterCrop(224),
				transforms.ToTensor(),
				transforms.Lambda(lambda x: x.mul(255)),
				normalize,
				])

	train_dataset = dataset_class(
				transform=train_trans,
				partition='train',
				class_file=opts.class_file,
				imgs_path_file=opts.imgs_path_file,
				image_dir=opts.image_dir,
				class_split=opts.class_split)

	train_loader = torch.utils.data.DataLoader(
					train_dataset,
					batch_size= opts.batchsize,
					shuffle=True,
					num_workers=opts.workers,
					pin_memory=True)
	
	
	test_dataset = dataset_class(
				transform=test_trans,
				partition='test',
				class_file=opts.class_file,
				imgs_path_file=opts.imgs_path_file,
				image_dir=opts.image_dir,
				class_split=opts.class_split)

	test_loader = torch.utils.data.DataLoader(
					test_dataset,
					batch_size= opts.batchsize,
					shuffle=False,
					num_workers=opts.workers,
					pin_memory=True)

	return train_loader, test_loader

if __name__ == '__main__':

	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
										std=[0.229, 0.224, 0.225])	

	# preparing the training loader
	train_dataset = MyDataset(
			transforms.Compose([
				transforms.Resize(256),
				transforms.CenterCrop(256),
				transforms.RandomCrop(224), # random crop within the center crop 
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				normalize,
			]),
			partition='train',
			class_file=opts.class_file,
			imgs_path_file=opts.imgs_path_file,
			image_dir=opts.image_dir,
			class_split=100)

	train_batch_sampler = MyBatchSampler(
						class_indices = train_dataset.class_indices,
						batch_size = opts.batchsize,
						same_class_num = opts.sameinclass)

	train_loader = torch.utils.data.DataLoader(
					train_dataset,
					batch_sampler=train_batch_sampler,
					#num_workers=opts.workers,
					pin_memory=True)
	print('Training loader prepared.')

	print(len(train_loader.batch_sampler.class_indices))
	train_dataset.class_indices['ciab'] = 1
	print(len(train_loader.batch_sampler.class_indices))
