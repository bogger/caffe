#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void inflate_gpu_kernel(const int n, const Dtype* data_bottom,
    const int height, const int width, const int inflate_size_h, const int inflate_size_w,
    const Dtype placeholder, const Dtype scale,
    const int inflate_rate_h, const int inflate_rate_w,
    Dtype* data_top) {
  CUDA_KERNEL_LOOP(index, n) {
	  int w_out = index % inflate_size_w;
	  int h_index = index / inflate_size_w;
	  int h_out = h_index % inflate_size_h;
	  int channel_out = h_index / inflate_size_h;

	  data_top[index] = ((w_out % inflate_rate_w == 0)&&(h_out % inflate_rate_h == 0))
			  	  	  	  ?data_bottom[(channel_out * height + h_out/inflate_rate_h)*width + w_out/inflate_rate_w] * scale
			  	  	  	  :placeholder;

  }
}

template <typename Dtype>
__global__ void shrink_gpu_kernel(const int n, const Dtype* data_top,
	    const int height, const int width, const int inflate_size_h, const int inflate_size_w,
	    const Dtype placeholder, const Dtype scale,
	    const int inflate_rate_h, const int inflate_rate_w,
	    Dtype* data_bottom) {
  CUDA_KERNEL_LOOP(index, n) {
    int w_bt = index % width;
    int h_index = index/width;
    int h_bt = h_index % height;
    int channel_bt = h_index/height;

    data_bottom[index] = data_top[(channel_bt*inflate_size_h + h_bt*inflate_rate_h)*inflate_size_w + w_bt*inflate_rate_w] * scale;
  }
}

template <typename Dtype>
void InflationLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  Forward_cpu(bottom, top);
	int num_kernels = channels_ * inflate_size_h_ * inflate_size_w_;
	const Dtype* data_bottom = bottom[0]->gpu_data();
	Dtype* data_top = (*top)[0]->mutable_gpu_data();

	for (int n = 0; n < bottom[0]->num(); n++){
		inflate_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
	                             CAFFE_CUDA_NUM_THREADS>>>(
	      num_kernels, data_bottom + bottom[0]->offset(n),
	      height_, width_, inflate_size_h_, inflate_size_w_,
	      placeholder_, scale_,
	      inflate_rate_h_, inflate_rate_w_,
	      data_top + (*top)[0]->offset(n)
	    );
		CUDA_POST_KERNEL_CHECK;
	}
}

template <typename Dtype>
void InflationLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
//  Backward_cpu(top, propagate_down, bottom);
	int num_kernels = channels_ * height_ * width_; //launch as many threads as the pixel number in the bottom data
	Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
	const Dtype* top_diff = top[0]->gpu_diff();

	caffe_gpu_set<Dtype>((*bottom)[0]->count(), Dtype(0.), bottom_diff);

	for (int n = 0; n < (*bottom)[0]->num(); n++){
		shrink_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
								 CAFFE_CUDA_NUM_THREADS>>>(
		  num_kernels, top_diff + top[0]->offset(n),
		  height_, width_, inflate_size_h_, inflate_size_w_,
		  placeholder_, scale_,
		  inflate_rate_h_, inflate_rate_w_,
		  bottom_diff + (*bottom)[0]->offset(n)
		);
		CUDA_POST_KERNEL_CHECK;
	}

}


INSTANTIATE_CLASS(InflationLayer);

}  // namespace caffe
