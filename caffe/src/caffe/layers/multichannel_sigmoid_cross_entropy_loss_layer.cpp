#include <algorithm>
#include <cfloat>
#include <vector>

#include <opencv2/opencv.hpp>

#include "caffe/layers/multichannel_sigmoid_cross_entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MultiChannelSigmoidCrossEntropyLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    LossLayer<Dtype>::LayerSetUp(bottom, top);

    CHECK(bottom[0]->channels()<=31) << "Currently only support at most 31 bit encoding!";
    CHECK(bottom[1]->channels()==1) << "Label only support 1 channel for now!";

    sigmoid_bottom_vec_.clear();
    sigmoid_bottom_vec_.push_back(bottom[0]);
    sigmoid_top_vec_.clear();
    sigmoid_top_vec_.push_back(sigmoid_output_.get());
    sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);
}

template <typename Dtype>
void MultiChannelSigmoidCrossEntropyLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    LossLayer<Dtype>::Reshape(bottom, top);
    // by simbaforrest, now bottom[1], i.e., target, will have less count than bottom[0]
    CHECK_EQ(bottom[0]->width()*bottom[0]->height(), bottom[1]->width()*bottom[1]->height()) <<
    "MULTICHANNEL_SIGMOID_CROSS_ENTROPY_LOSS layer inputs must have the same spatial dimension.";
    sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);
}

template <typename Dtype>
void MultiChannelSigmoidCrossEntropyLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    // The forward pass computes the sigmoid outputs.
    sigmoid_bottom_vec_[0] = bottom[0];
    sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
    // Compute the loss (negative log likelihood)
    const int count = bottom[0]->count();
    const int num = bottom[0]->num();
    // Stable version of loss computation from input data
    const Dtype* input_data = bottom[0]->cpu_data();
    const Dtype* target = bottom[1]->cpu_data();
    const int nchannels = bottom[0]->channels(); //target is 1-channel, but we treat individual bits as target channels
    Dtype loss_pos = 0;
    Dtype loss_neg = 0;

    const unsigned int ignored_bit = (1 << 31);
    int WxH = count / (num * nchannels);
    for (int i = 0; i < num; ++i) {//each image
        const int offset_label = i*WxH;
        // Compute channel-wise sigmoid loss
        for(int k = 0; k < nchannels; ++k) {//each input channel
            const int offset_input = (i*nchannels+k)*WxH;
            for (int j = 0; j < WxH; j ++) {//each pixel
                const Dtype& input_j_k = input_data[offset_input + j];
                const unsigned int* ptr = reinterpret_cast<const unsigned int*>(target+offset_label+j);
                const bool ignored = (*ptr) & ignored_bit;
                if(ignored) continue;
                const bool label_j_k = (*ptr) & (1 << k);
                if (label_j_k) {
                    loss_pos -= input_j_k * (1 - (input_j_k >= 0)) -
                    log(1 + exp(input_j_k - 2 * input_j_k * (input_j_k >= 0)));
                } else {
                    loss_neg -= input_j_k * (0 - (input_j_k >= 0)) -
                    log(1 + exp(input_j_k - 2 * input_j_k * (input_j_k >= 0)));
                }
            }//j
        }//k
    }//i
    top[0]->mutable_cpu_data()[0] = (loss_pos * 1 + loss_neg) / num;
}

template <typename Dtype>
void MultiChannelSigmoidCrossEntropyLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    if (propagate_down[1]) {
        LOG(FATAL) << this->type()
                   << " Layer cannot backpropagate to label inputs.";
    }
    if (propagate_down[0]) {
    // Get the dimensions
    const int count = bottom[0]->count();
    const int num = bottom[0]->num();
    const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();
    const Dtype* target = bottom[1]->cpu_data();
    const int nchannels = bottom[0]->channels(); //target is 1-channel, but we treat individual bits as target channels
    // Decode the multi-channel target
    const unsigned int ignored_bit = (1 << 31);
    Dtype* pNewTarget = new Dtype[count];
    int WxH = count / (num * nchannels);
    for(int i=0; i<num; ++i) {
        const int offset_label = i*WxH;
        for(int k=0; k<nchannels; ++k) {
            const int offset_input = (i*nchannels+k)*WxH;
            for (int j = 0; j < WxH; j ++) {//each pixel
                const unsigned int* ptr = reinterpret_cast<const unsigned int*>(target+offset_label+j);
                const bool ignored = (*ptr) & ignored_bit;
                if(ignored) {
                    pNewTarget[offset_input + j] = sigmoid_output_data[offset_input + j]; //pre-fill target with pred so that gradient = pred-target = 0
                } else {
                    pNewTarget[offset_input + j] = (((*ptr) & (1 << k)) > 0) ? 1 : 0;
                }
            }//j
        }//k
    }//i
    // Compute the diff
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_sub(count, sigmoid_output_data, pNewTarget, bottom_diff);
    const Dtype loss_weight = top [0]->cpu_diff()[0];
    caffe_scal(count, loss_weight / num, bottom_diff);
    delete[] pNewTarget;
    }//propagate_down[0]
}

#ifdef CPU_ONLY
STUB_GPU_BACKWARD(MultiChannelSigmoidCrossEntropyLossLayer, Backward);
#endif

INSTANTIATE_CLASS(MultiChannelSigmoidCrossEntropyLossLayer);
REGISTER_LAYER_CLASS(MultiChannelSigmoidCrossEntropyLoss);

}  // namespace caffe
