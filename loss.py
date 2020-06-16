import os, sys, time
import math
import random
import numpy as np

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

class JRS(nn.Module):
	def __init__(self, kernel_muls, kernel_nums, fix_sigmas):
		super(JRS, self).__init__()
		self.kernel_muls = kernel_muls
		self.kernel_nums = kernel_nums
		self.fix_sigmas = fix_sigmas

	def euclidean_dist(self, x, y):
		b = x.size(0)
		xx = torch.pow(x, 2).sum(1, keepdim=True).expand(b, b)
		yy = torch.pow(y, 2).sum(1, keepdim=True).expand(b, b).t()
		dist = xx+yy-2*torch.mm(x, y.t())
		return dist

	def guassian_kernel(self, source, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
		n = source.size(0)
		L2_distance = self.euclidean_dist(source, source)
		if fix_sigma:
			bandwidth = fix_sigma
		else:
			bandwidth = torch.sum(L2_distance.data) / (n**2-n)
		bandwidth /= kernel_mul ** (kernel_num//2)
		bandwidth_list = [bandwidth*(kernel_mul**i) for i in range(kernel_num)]
		kernel_val = [torch.exp(-L2_distance/bandwidth_temp) for bandwidth_temp in bandwidth_list]
		return sum(kernel_val)/len(kernel_val)

	def forward(self, source_list, target):
		b = source_list[0].size(0)
		layer_num = len(source_list)
		joint_kernels = None
		for i in range(layer_num):
			source = source_list[i]
			kernel_mul = self.kernel_muls[i]
			kernel_num = self.kernel_nums[i]
			fix_sigma = self.fix_sigmas[i]
			kernels = self.guassian_kernel(source, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
			if joint_kernels is not None:
				joint_kernels = joint_kernels*kernels
			else:
				joint_kernels = kernels

		class_mat = target.repeat(target.size(0), 1)
		same_class = torch.eq(class_mat, class_mat.t())
		anti_class = same_class.clone()
		anti_class = anti_class == 0

		pos_samples = torch.masked_select(joint_kernels, same_class)
		neg_samples = torch.masked_select(joint_kernels, anti_class)

		loss = neg_samples.mean()

		return loss

class CosineSoftmax(nn.Module):
	def __init__(self, scale, M, dim, cN):
		super(CosineSoftmax, self).__init__()
		self.scale = scale
		self.M = M
		self.cN = cN
		self.fc = Parameter(torch.Tensor(dim, cN))

		torch.nn.init.kaiming_uniform_(self.fc, a=math.sqrt(5))

	def getSim(self, inputs):
		centers = F.normalize(self.fc, p=2, dim=0)
		simClass = inputs.matmul(centers)
		return simClass

	def forward(self, simClass, target):
		marginM = torch.zeros(target.size(0), self.cN).cuda()
		marginM = marginM.scatter_(1, target.view(-1,1), self.M)

		loss = F.cross_entropy(self.scale*(simClass-marginM), target)
		return loss


class OurLoss(nn.Module):
	def __init__(self, scale, M, dim, cN, kernel_muls, kernel_nums, fix_sigmas, alpha):
		super(OurLoss, self).__init__()
		self.soft = CosineSoftmax(scale, M, dim, cN)
		self.jrs = JRS(kernel_muls, kernel_nums, fix_sigmas)
		self.alpha = alpha
		self.cN = cN

	def forward(self, feats, emb, target):
		# feats is a list
		simClass = self.soft.getSim(emb)
		jrsloss = self.jrs(feats+[emb,simClass], target)
		
		softloss = self.soft(simClass, target)
		return softloss+self.alpha*jrsloss
