#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void InflationLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {

	InflationParameter inflation_param = this->layer_param_.inflation_param();
	CHECK(!inflation_param.has_inflate_size() !=
			!(inflation_param.has_inflate_size_h() && inflation_param.has_inflate_size_w()))
		<< "Inflated image size is inflate_size OR inflate_size_h and inflate_size_w; not both";
	CHECK(inflation_param.has_inflate_size() ||
			(inflation_param.has_inflate_size_h() && inflation_param.has_inflate_size_w()))
		<< "For non-square inflation both h and w are required.";

	if (inflation_param.has_inflate_size()){
		inflate_size_h_ = inflate_size_w_ = inflation_param.inflate_size();
	}else{
		inflate_size_h_ = inflation_param.inflate_size_h();
		inflate_size_w_ = inflation_param.inflate_size_w();
	}

	placeholder_ = inflation_param.placeholder();
	scale_ = inflation_param.scale();


	CHECK(inflate_size_h_>bottom[0]->height())
		<< "Inflated height "<<inflate_size_h_<<" must be larger than input height "<<bottom[0]->height();
	CHECK(inflate_size_w_>bottom[0]->width())
			<< "Inflated width "<<inflate_size_w_<<" must be larger than input width "<<bottom[0]->width();

	inflate_rate_h_ = inflate_size_h_/bottom[0]->height();
	inflate_rate_w_ = inflate_size_w_/bottom[0]->width();

}

template <typename Dtype>
void InflationLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  (*top)[0]->Reshape(
		bottom[0]->num(), channels_,
		inflate_size_h_, inflate_size_w_
  );

}

template <typename Dtype>
void InflationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
//  const Dtype* bottom_data = bottom[0]->cpu_data();
//  Dtype* top_data = (*top)[0]->mutable_cpu_data();
//  for (int n = 0; n < bottom[0]->num(); ++n) {
//    im2col_cpu(bottom_data + bottom[0]->offset(n), channels_, height_,
//        width_, kernel_h_, kernel_w_, pad_h_, pad_w_,
//        stride_h_, stride_w_, top_data + (*top)[0]->offset(n));
//  }
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* top_data = (*top)[0]->mutable_cpu_data();

	for (int n = 0; n < bottom[0]->num(); ++n){

		//get initial point
		const Dtype* ptr_bottom = bottom_data + bottom[0]->offset(n);
		Dtype* ptr_top = top_data + (*top)[0]->offset(n);

		for (int ch = 0; ch < channels_; ++ch){
			//inflate per channel
			for (int h = 0; h < inflate_size_h_; ++h){
				for (int w = 0; w < inflate_size_w_; ++w){
					ptr_top[h*inflate_size_w_ + w] = ((h%inflate_rate_h_==0)&&(w%inflate_rate_w_==0))
							?ptr_bottom[(h/inflate_rate_h_)*width_ + (w/inflate_rate_w_)] * scale_
							:placeholder_;
				}
			}

			//offset base pointers
			ptr_bottom += height_*width_;
			ptr_top += inflate_size_h_*inflate_size_w_;
		}
	}
}

template <typename Dtype>
void InflationLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {

	const Dtype* top_diff = top[0]->cpu_diff();
	Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();

	caffe_set<Dtype>((*bottom)[0]->count(), Dtype(0.), bottom_diff);

	for (int n = 0; n < (*bottom)[0]->num(); ++n){
		//get initial point
		Dtype* ptr_bottom = bottom_diff + (*bottom)[0]->offset(n);
		const Dtype* ptr_top = top_diff + top[0]->offset(n);

		for (int ch = 0; ch < channels_; ++ch){

			//shrink gradient maps to input size
			for (int h = 0; h < height_; ++h){
				for (int w = 0; w < width_; ++w){
					ptr_bottom[h*width_ + w] = ptr_top[(h*inflate_rate_h_)*inflate_size_w_ + w*inflate_rate_w_] * scale_;
				}
			}

			//offset base pointers
			ptr_bottom += height_*width_;
			ptr_top += inflate_size_h_*inflate_size_w_;
		}
	}
}

#ifdef CPU_ONLY
STUB_GPU(InflationLayer);
#endif

INSTANTIATE_CLASS(InflationLayer);

}  // namespace caffe
