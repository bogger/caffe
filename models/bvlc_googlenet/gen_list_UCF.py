import os
import glob
import sys
im_dir = '/media/researchshare/linjie/data/UCF-101_images/'
#out_dir = '/media/researchshare/linjie/data/snapchat/features/cnn_resize/'
fds = os.listdir(im_dir)
#out_dir = 'output/c3d/'
#sav_name='UCF_list_frm.txt'
#prefix_name='prototxt/snapchat_list_prefix.txt'
#prefix_file = open(prefix_name,'w')
interval = 5
if len(sys.argv)<2:
	print "please input stage"
	exit()

stage=sys.argv[1]
list_path = '/home/a-linjieyang/work/video_caption/ucfTrainTestlist/%slist01.txt' % stage
invalid_n=0
sav_name = 'UCF_list_frm_%s.txt' % stage
list_file = open(sav_name,'w')

with open(list_path,'r') as f:
	for line in f:
		content = line.split()
		vid_fd = content[0][:-4]
		if len(content)==1:
			label=0
		else:
			label = int(content[1])-1
		images = os.listdir(im_dir+vid_fd)
		n_im = len(images)
		if n_im==0:
			invalid_n+=1
		for i in xrange(1,n_im+1,interval):
			list_file.write('%s%s/%06d.jpg %d\n' %  (im_dir, vid_fd, i, label))
		#prefix_file.write('%s%s/%06d\n' % (out_dir, fd, i))
		#if not os.path.exists(out_dir + fd):
		#	os.mkdir(out_dir + fd)
list_file.close()
print invalid_n
#prefix_file.close()
