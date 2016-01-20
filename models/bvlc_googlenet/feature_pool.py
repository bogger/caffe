import cPickle
import os
import numpy as np
list_name ='snapchat_list_frm.txt'
feat_name='/media/researchshare/linjie/data/snapchat/features/googlenet.bin'
with open(feat_name, 'rb') as fb:
	feats_all = cPickle.load(fb)
	print feats_all.shape
fd_pre=''
feat_dim=1024
im_dir = '/media/researchshare/linjie/data/snapchat/images/'
feat_dir = '/media/researchshare/linjie/data/snapchat/features/c3d/'
fds = os.listdir(feat_dir)
vid_n = len(fds)
agg_feats=np.zeros((vid_n,feat_dim), dtype=np.float32)
feats = np.zeros((1,feat_dim),dtype=np.float32)
with open(list_name, 'r') as f:
	c=0
	for i, line in enumerate(f):
		im_path = line.split()[0]
		fd = im_path.split('/')[-2]
		if not os.path.exists(feat_dir+fd):
			continue
		if fd_pre != '' and fd != fd_pre:
			agg_feats[c,:] = feats
			c+=1
			feats = feats_all[i,:]
		else:
			feats = np.maximum(feats, feats_all[i,:])
		fd_pre = fd
	#print im_path
	#exit()
	agg_feats[c,:] = feats
	c+=1
	
	#consistency check
	if c != vid_n:
		print 'video number not match! c is %d, vid_n is %d' % (c,vid_n)
#vid_n = c
#find nearest
import shutil
top_n =5
sample_r = 100
src_dir = '/media/researchshare/linjie/data/snapchat/video/'
sav_dir = '/media/researchshare/linjie/data/snapchat/similar_googlenet/'
similar = np.zeros((vid_n, top_n), dtype=np.int32)
for i in xrange(vid_n):
	dist = np.linalg.norm(agg_feats - agg_feats[i,:], axis=1)
	index = np.argsort(dist)
	similar[i,:] = index[1:top_n+1]
for i in xrange(0, vid_n, sample_r):#sample some videos
	if not os.path.exists(sav_dir+fds[i]):
		os.makedirs(sav_dir+fds[i])
	shutil.copy(src_dir+fds[i]+'.mp4', sav_dir+fds[i]+'/source.mp4')
	for j in xrange(top_n):
		shutil.copy(src_dir+fds[similar[i,j]]+'.mp4', sav_dir+fds[i]+'/top'+str(j)+'.mp4')

