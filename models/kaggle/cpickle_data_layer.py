import caffe
import numpy as np
import scipy.io as sio
import time
import cPickle
from random import shuffle
import json

class CPickleDataLayer(caffe.Layer):
	def setup(self,bottom,top):
		if len(bottom) >= 1:
			raise Exception("This is a data layer, it takes no bottom blob")
		self.json_param = json.loads(self.param_str)
		self.data_file = self.json_param['source']
		self.batch_size = self.json_param['batch_size']
		#self.channels = 20
		if len(top)!=2:
			raise Exception("top size can only be 2: input and label")
		self.data = cPickle.load(open(self.data_file))
		#self.data = tmp['data']
		self.data_cursor = iter(self.data)
		#use a data point to initialize param
		sample_input = self.data[0]['input']
		sample_label = self.data[0]['label']
		data_shape = sample_input.shape
		if len(data_shape) == 2:
			self.height = 1#data_shape[2]
		elif len(data_shape)==3:
			self.height = data_shape[2]
		else:
			raise Exception("data shape must be at least two-dimensional")
		self.width = data_shape[1]
		
		self.channels = data_shape[0]
		self.label_size = 1#sample_label.shape[0]
	def get_next_sample(self):
		try:
			next_sample = self.data_cursor.next()
		except StopIteration, e:
			#shuffle		
			shuffle(self.data)
			self.data_cursor = iter(self.data)
			next_sample = self.data_cursor.next()
		return next_sample
	def reshape(self, bottom, top):
		top[0].reshape(self.batch_size,self.channels,self.width,self.height)
		top[1].reshape(self.batch_size,self.label_size)
		self.data_buffer = np.empty((self.batch_size,self.channels,self.width,self.height),
			dtype=np.float32)
	def forward(self,bottom,top):
		labels = np.empty((self.batch_size,self.label_size),dtype=np.float32)
		for n in xrange(self.batch_size):
			sample = self.get_next_sample()
			if self.height>1:
				self.data_buffer[n,:,:,:] = sample['input']
			else:
				self.data_buffer[n,:,:,0] = sample['input']
				
			labels[n,:] = sample['label']/10#normalize
		top[0].data[:] = self.data_buffer[:]
		top[1].data[:] = labels[:]

