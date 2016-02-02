import os
import glob
im_dir = '/media/researchshare/linjie/data/snapchat/images/'
#out_dir = '/media/researchshare/linjie/data/snapchat/features/cnn_resize/'
fds = os.listdir(im_dir)
#out_dir = 'output/c3d/'
sav_name='snapchat_list_frm.txt'
#prefix_name='prototxt/snapchat_list_prefix.txt'
list_file = open(sav_name,'w')
#prefix_file = open(prefix_name,'w')
interval = 5
invalid_n=0
for fd in fds:
	images = os.listdir(im_dir+fd)
	n_im = len(images)
	if n_im==0:
		invalid_n+=1
	for i in xrange(1,n_im+1,interval):
		list_file.write('%s%s/%06d.jpg 0\n' %  (im_dir, fd, i))
		#prefix_file.write('%s%s/%06d\n' % (out_dir, fd, i))
		#if not os.path.exists(out_dir + fd):
		#	os.mkdir(out_dir + fd)
list_file.close()
print invalid_n
#prefix_file.close()
