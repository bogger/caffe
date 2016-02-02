import cPickle 
import pandas as pd
import os
import numpy as np
import sys
stage = sys.argv[1]
list_name ='UCF_list_frm_%s.txt' % stage

sav_dir = '/media/researchshare/linjie/data/UCF-101_features/'
feat_name='%sgooglenet_%s.bin' % (sav_dir, stage)
with open(feat_name, 'rb') as fb:
	feats_all = cPickle.load(fb)
	print feats_all.shape
fd_pre=''
feat_dim=1024
pool_type='mean'
im_dir = '/media/researchshare/linjie/data/UCF-101_images/'
#feat_dir = '/media/researchshare/linjie/data/UCF/images/'
#fds = os.listdir(im_dir)
#vid_n = len(fds)
#agg_feats=np.zeros((vid_n,feat_dim), dtype=np.float32)
#feats = np.zeros((1,feat_dim),dtype=np.float32)
#id_pre=0

info = pd.read_table(list_name,sep=' ')
feat_n =info.shape[0]
info.columns = ['path','label']
info['feat'] = pd.Series([feats_all[i,:] for i in xrange(feat_n)])
info['fd'] = pd.Series([path.split('/')[-2] for path in info['path']])
if pool_type=='mean':
	pooled_feats = info.groupby('fd').apply(lambda x: np.mean(x['feat'], axis=0))
else:
	pooled_feats = info.groupby('id').apply(lambda x: np.amax(x['feat'], axis=0))
#c=pooled_feats.shape[0]
# convert series of numpy array to numpy matrix
pooled_feats = np.vstack(pooled_feats)
#consistency check
print pooled_feats.shape
#if c != vid_n:
#	print 'video number not match! c is %d, vid_n is %d' % (c,vid_n)
with open('%sgooglenet_pooled_%s' % (sav_dir, stage),'wb') as fb:
	cPickle.dump(pooled_feats,fb,protocol=cPickle.HIGHEST_PROTOCOL)

