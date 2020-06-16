import argparse

def get_parser():
	parser = argparse.ArgumentParser(description='model parameters')
	
	# CUB args
	# data
	parser.add_argument('--data_name', default='CUB')
	parser.add_argument('--data_path', default='../CUB_200_2011')
	parser.add_argument('--class_file', default='image_class_labels.txt')
	parser.add_argument('--imgs_path_file', default='images.txt')
	parser.add_argument('--image_dir', default='images')
	parser.add_argument('--class_split', default=100,type=int)
	
	# training
	parser.add_argument('--seed', default=2019,type=int)
	parser.add_argument('--workers', default=4, type=int)
	parser.add_argument('--checkpoint', default='./ckpt/', type=str)
	parser.add_argument('--maxCkpt', default=3, type=int)
	parser.add_argument('--restore', default='', type=str)
	
	parser.add_argument('--lr', default=0.0001,type=float)
	parser.add_argument('--lr_img', default=0.0001,type=float)
	parser.add_argument('--lr_cent', default=0.01,type=float)
	parser.add_argument('--lr_update', default=20, type=int)
	parser.add_argument('--decay_rate', default=0.1, type=float)

	parser.add_argument('--eps', default=0.01, type=float)
	parser.add_argument('--weight_decay', default=1e-4, type=float)
	parser.add_argument('--dropout', default=0.1,type=float)
	parser.add_argument('--gradclip', default=-1.0, type=float)
	parser.add_argument('--freeze_BN', default=True, type=bool)

	parser.add_argument('--epochs', default=50,type=int)
	parser.add_argument('--valfreq', default=1,type=int)

	# cosine softmax hyper
	parser.add_argument('--margin', default=0.1,type=float)
	parser.add_argument('--scale', default=20.0,type=float)

	# JRS hyper
	parser.add_argument('--kernelMuls', default=[2.0,2.0,2.0],type=list)
	parser.add_argument('--kernelNums', default=[3,3,1],type=list)
	parser.add_argument('--sigmas', default=[None,None,None],type=list)
	parser.add_argument('--alpha', default=1.0,type=float)

	# batch
	parser.add_argument('--embDim', default=512,type=int)
	parser.add_argument('--batchsize', default=100,type=int)

	parser.add_argument('--gpu', default='2',type=str)
	
	return parser


