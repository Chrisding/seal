#include <algorithm>
#include <cfloat>
#include <vector>

#include <opencv2/opencv.hpp>

#include "caffe/layers/multichannel_reweighted_sigmoid_cross_entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MultiChannelReweightedSigmoidCrossEntropyLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    LossLayer<Dtype>::LayerSetUp(bottom, top);
    CHECK( bottom[0]->channels() < (int)(bottom[1]->channels()*32) )
	<< "The number of data channels must be smaller than label channels!";// Modified by Zhiding

    sigmoid_bottom_vec_.clear();
    sigmoid_bottom_vec_.push_back(bottom[0]);
    sigmoid_top_vec_.clear();
    sigmoid_top_vec_.push_back(sigmoid_output_.get());
    sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);
}

template <typename Dtype>
void MultiChannelReweightedSigmoidCrossEntropyLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    LossLayer<Dtype>::Reshape(bottom, top);
    // by simbaforrest, now bottom[1], i.e., target, will have less count than bottom[0]
    CHECK_EQ(bottom[0]->width()*bottom[0]->height(), bottom[1]->width()*bottom[1]->height()) <<
    "MULTICHANNEL_REWEIGHTED_SIGMOID_CROSS_ENTROPY_LOSS layer inputs must have the same spatial dimension.";
    sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);
}

template <typename Dtype>
void MultiChannelReweightedSigmoidCrossEntropyLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    // The forward pass computes the sigmoid outputs.
    sigmoid_bottom_vec_[0] = bottom[0];
    sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
    // Compute the loss (negative log likelihood)
    const int count = bottom[0]->count();// Volume size
    const int num = bottom[0]->num();// Batch size
    // Stable version of loss computation from input data
    const Dtype* input_data = bottom[0]->cpu_data();
    const Dtype* target = bottom[1]->cpu_data();
    const int data_channels = bottom[0]->channels();
	const int label_channels = bottom[1]->channels();// Target blob is n-channel, where n*32 > data_channel. We treat each float bit of the blob as a target channel.
	const int label_channels_use = data_channels / (int)(32) + 1;// Number of effective label channels in use.
    Dtype loss_pos = 0;
    Dtype loss_neg = 0;

    const unsigned int ignored_bit = (1 << 31);
    int WxH = count / (num * data_channels);
	const int offset_img_ignore = (label_channels - 1) * WxH;// Precompute the offset of the label channel containing ignore bit in each image.
    for (int i = 0; i < num; ++i) {// image index
        const int offset_batch_label = i*label_channels*WxH;// Precompute the global offset of each label image.
		const int offset_ignore = offset_batch_label + offset_img_ignore; // Precompute the global offset of the label channel containing ignore bit.
        // Count class-agnostic edge pixels
        Dtype count_pos = 0;
        Dtype count_neg = 0;
		for (int j = 0; j < WxH; ++j) {// (H, W) pixel index
			const unsigned int* ptr_ignore = reinterpret_cast<const unsigned int*>(target + offset_ignore + j);
			const bool ignored = (*ptr_ignore) & ignored_bit;
			if (ignored) continue;
			bool flag_pos = false;
			for (int k = 0; k < label_channels_use; ++k){// Channel index
				const unsigned int* ptr = reinterpret_cast<const unsigned int*>(target + offset_batch_label + k*WxH + j);
				if (*ptr != 0) {
					flag_pos = true;
					break;
				}
			}
			if (flag_pos) count_pos++;
			else count_neg++;
		}
        const Dtype count_all = count_pos + count_neg;
        // Compute channel-wise sigmoid loss and reweight with class-agnostic edge pixel count
        for(int k = 0; k < data_channels; ++k) {// Channel index
            Dtype temp_loss_pos = 0;
            Dtype temp_loss_neg = 0;
            const int offset_input = (i*data_channels+k)*WxH;
			const int quotient_label = k / (int)(32);// Precompute the corresponding label channel
			const int remainder_label = k % 32;// Precompute the corresponding bit position in a label channel
			const unsigned int true_bit = (1 << remainder_label);// Precompute the 32-bit label with one specific bit being true
			const int offset_label = offset_batch_label + quotient_label*WxH;// Precompute the global offset of the label channel containing interested bit
            for (int j = 0; j < WxH; j ++) {// (H, W) pixel index
                const Dtype& input_j_k = input_data[offset_input + j];
				const unsigned int* ptr_ignore = reinterpret_cast<const unsigned int*>(target + offset_ignore + j);
                const bool ignored = (*ptr_ignore) & ignored_bit;
                if(ignored) continue;
				const unsigned int* ptr = reinterpret_cast<const unsigned int*>(target + offset_label + j);
				const bool label_j_k = (*ptr) & true_bit;
                if (label_j_k) {
                    temp_loss_pos -= input_j_k * (1 - (input_j_k >= 0)) -
                    log(1 + exp(input_j_k - 2 * input_j_k * (input_j_k >= 0)));
                } else {
                    temp_loss_neg -= input_j_k * (0 - (input_j_k >= 0)) -
                    log(1 + exp(input_j_k - 2 * input_j_k * (input_j_k >= 0)));
                }
            }//j
            if(count_all!=0) {
                loss_pos += temp_loss_pos * (1.0 * count_neg / count_all);
                loss_neg += temp_loss_neg * (1.0 * count_pos / count_all);
            }
        }//k
    }//i
    top[0]->mutable_cpu_data()[0] = (loss_pos * 1 + loss_neg) / num;
}

template <typename Dtype>
void MultiChannelReweightedSigmoidCrossEntropyLossLayer<Dtype>::Backward_cpu(
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
    const int data_channels = bottom[0]->channels(); //target is 1-channel, but we treat individual bits as target channels
	const int label_channels = bottom[1]->channels();// Target blob is n-channel, where n*32 > data_channel. We treat each float bit of the blob as a target channel.
	const int label_channels_use = data_channels / (int)(32) + 1;// Number of effective label channels in use.
    // Decode the multi-channel target
    const unsigned int ignored_bit = (1 << 31);
    Dtype* pNewTarget = new Dtype[count];
    int WxH = count / (num * data_channels);
	const int offset_img_ignore = (label_channels - 1) * WxH;// Precompute the offset of the label channel containing ignore bit in each image.
    for(int i=0; i<num; ++i) {
		const int offset_batch_label = i*label_channels*WxH;// Precompute the global offset of each label image.
		const int offset_ignore = offset_batch_label + offset_img_ignore; // Precompute the global offset of the label channel containing ignore bit.
        for(int k=0; k<data_channels; ++k) {
			const int offset_input = (i*data_channels + k)*WxH;
			const int quotient_label = k / (int)(32);// Precompute the corresponding label channel
			const int remainder_label = k % 32;// Precompute the corresponding bit position in a label channel
			const unsigned int true_bit = (1 << remainder_label);// Precompute the 32-bit label with one specific bit being true
			const int offset_label = offset_batch_label + quotient_label*WxH;// Precompute the global offset of the label channel containing interested bit
            for (int j = 0; j < WxH; j ++) {//each pixel
                const unsigned int* ptr_ignore = reinterpret_cast<const unsigned int*>(target + offset_ignore + j);
                const bool ignored = (*ptr_ignore) & ignored_bit;
                if(ignored) {
                    pNewTarget[offset_input + j] = -100; //ignore label
                } else {
					const unsigned int* ptr = reinterpret_cast<const unsigned int*>(target + offset_label + j);
                    pNewTarget[offset_input + j] = (((*ptr) & true_bit) > 0) ? 1 : 0;
                }
            }//j
        }//k
    }//i
    // Compute the diff
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_sub(count, sigmoid_output_data, pNewTarget, bottom_diff);
    // Reweight the diff
    for (int i = 0; i < num; ++i) {//each image
        // Count class-agnostic edge pixels
        Dtype count_pos = 0;
        Dtype count_neg = 0;
		const int offset_batch_label = i*label_channels*WxH;// Precompute the global offset of each label image.
		const int offset_ignore = offset_batch_label + offset_img_ignore; // Precompute the global offset of the label channel containing ignore bit.
        for (int j = 0; j < WxH; j ++) {
            const unsigned int* ptr_ignore = reinterpret_cast<const unsigned int*>(target + offset_ignore + j);
            const bool ignored = (*ptr_ignore) & ignored_bit;
            if(ignored) continue;
			bool flag_pos = false;
			for (int k = 0; k < label_channels_use; ++k){// Channel index
				const unsigned int* ptr = reinterpret_cast<const unsigned int*>(target + offset_batch_label + k*WxH + j);
				if (*ptr != 0) {
					flag_pos = true;
					break;
				}
			}
			if (flag_pos) count_pos++;
			else count_neg++;
        }
        const Dtype count_all = count_pos + count_neg;
        Dtype weight_pos = 0;
        Dtype weight_neg = 0;
        if(count_all!=0) {
            weight_pos = 1.0 * count_neg / count_all;
            weight_neg = 1.0 * count_pos / count_all; // Added "1.0". Modified by Zhiding
        }
        // Reweight the diff with class-agnostic edge pixel count
        for(int k = 0; k < data_channels; ++k) {//each channel
            const int offset = (i*data_channels+k)*WxH;
            for (int j = 0; j < WxH; j ++) {
                const int pix = offset + j;
                const Dtype& label_j_k = pNewTarget[pix];
                if (label_j_k == 1) {
                    bottom_diff[pix] *= weight_pos;
                } else if (label_j_k == 0) {
                    bottom_diff[pix] *= weight_neg;
                } else if (label_j_k == -100) {
                    bottom_diff[pix] = 0; //pixels with ignore label should not have derivative
                }
            }//j
        }//k
    }//i
    delete[] pNewTarget;
    const Dtype loss_weight = top [0]->cpu_diff()[0];
    caffe_scal(count, loss_weight / num, bottom_diff);
    }//propagate_down[0]
}

#ifdef CPU_ONLY
STUB_GPU_BACKWARD(MultiChannelReweightedSigmoidCrossEntropyLossLayer, Backward);
#endif

INSTANTIATE_CLASS(MultiChannelReweightedSigmoidCrossEntropyLossLayer);
REGISTER_LAYER_CLASS(MultiChannelReweightedSigmoidCrossEntropyLoss);

}  // namespace caffe
