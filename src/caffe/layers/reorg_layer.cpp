#include <vector>

#include "caffe/layers/reorg_layer.hpp"

namespace caffe {

template <typename Dtype>
void ReorgLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
  const ReorgParameter& reorg_param = this->layer_param_.reorg_param();
  stride_ = reorg_param.stride();
}

template <typename Dtype>
void ReorgLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not "
      "allow in-place computation.";

  vector<int> top_shape;
  top_shape.push_back(bottom[0]->shape(0));
  top_shape.push_back(bottom[0]->shape(1) * stride_ * stride_);
  top_shape.push_back(bottom[0]->shape(2) / stride_);
  top_shape.push_back(bottom[0]->shape(3) / stride_);

  top[0]->Reshape(top_shape);
  CHECK_EQ(top[0]->count(), bottom[0]->count());
  //LOG(INFO) << "shape:" << top[0]->shape_string();
}

template <typename Dtype>
void ReorgLayer<Dtype>::reorg_cpu(Dtype* input, int w, int h, int c, int batch, int stride, int forward, Dtype* output){
  int b, i, j, k;
  int out_c =  c / (stride * stride);
  
  for (b = 0; b < batch; ++ b){
    for (k = 0; k < c; ++ k){
      for (j = 0; j < h; ++ j){
	for (i = 0; i < w; ++ i){
          int in_index = i + w * (j + h * (k + c * b));
          int c2 = k % out_c;
          int offset = k / out_c;
          int w2 = i * stride + offset % stride;
          int h2 = j * stride + offset / stride;
          int out_index =  w2 + w * stride * (h2 + h * stride * (c2 + out_c * b));
          if (forward) output[out_index] = input[in_index];
	  else output[in_index] = input[out_index];
        }
      }
    }
  }
}

template <typename Dtype>
void ReorgLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Dtype* input_data = bottom[0]->mutable_cpu_data();
  Dtype* output_data = top[0]->mutable_cpu_data();
  reorg_cpu(input_data, bottom[0]->width(), bottom[0]->height(), bottom[0]->channels(), bottom[0]->num(), stride_, 0, output_data);
}

template <typename Dtype>
void ReorgLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if(propagate_down[0]) {
    Dtype* input_data = bottom[0]->mutable_cpu_data();
    Dtype* output_data = top[0]->mutable_cpu_data();
    reorg_cpu(input_data, bottom[0]->width(), bottom[0]->height(), bottom[0]->channels(), bottom[0]->num(), stride_, 1, output_data);
  }
}

INSTANTIATE_CLASS(ReorgLayer);
REGISTER_LAYER_CLASS(Reorg);

}  // namespace caffe
