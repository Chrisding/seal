#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/reweighted_sigmoid_cross_entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ReweightedSigmoidCrossEntropyLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    LossLayer<Dtype>::LayerSetUp(bottom, top);

    CHECK(bottom[0]->channels()==1) << "Input only support 1 channel for now!";
    CHECK(bottom[1]->channels()==1) << "Label only support 1 channel for now!";

    sigmoid_bottom_vec_.clear();
    sigmoid_bottom_vec_.push_back(bottom[0]);
    sigmoid_top_vec_.clear();
    sigmoid_top_vec_.push_back(sigmoid_output_.get());
    sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);
}

template <typename Dtype>
void ReweightedSigmoidCrossEntropyLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    LossLayer<Dtype>::Reshape(bottom, top);
    // by Zhiding
    CHECK_EQ(bottom[0]->width()*bottom[0]->height(), bottom[1]->width()*bottom[1]->height()) <<
    "SIGMOID_CROSS_ENTROPY_LOSS layer inputs must have the same spatial dimension.";
    sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);
}

template <typename Dtype>
void ReweightedSigmoidCrossEntropyLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    // The forward pass computes the sigmoid outputs.
    sigmoid_bottom_vec_[0] = bottom[0];
    sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
    // Compute the loss (negative log likelihood)
    // Stable version of loss computation from input data
    const int num = bottom[0]->num();
    const Dtype* input_data = bottom[0]->cpu_data();
    const Dtype* target = bottom[1]->cpu_data();
    Dtype loss_pos = 0;
    Dtype loss_neg = 0;
    Dtype temp_loss_pos = 0;
    Dtype temp_loss_neg = 0;
    Dtype count_pos = 0;
    Dtype count_neg = 0;
    const unsigned int ignored_bit = (1 << 31);
    int dim = bottom[0]->count() / bottom[0]->num();

    for (int i = 0; i < num; ++i) {
        temp_loss_pos = 0;
        temp_loss_neg = 0;
        count_pos = 0;
        count_neg = 0;

        const int offset = i*dim;
        for (int j = 0; j < dim; j ++) {
            const unsigned int* ptr = reinterpret_cast<const unsigned int*>(target+offset+j);
            const bool ignored = (*ptr) & ignored_bit;
            if(ignored) continue;
            if ((*ptr) != 0) {
                count_pos ++;
                temp_loss_pos -= input_data[offset + j] * (1 - (input_data[offset + j] >= 0)) -
                log(1 + exp(input_data[offset + j] - 2 * input_data[offset + j] * (input_data[offset + j] >= 0)));
            } else if ((*ptr) == 0) {
                count_neg ++;
                temp_loss_neg -= input_data[offset + j] * (0 - (input_data[offset + j] >= 0)) -
                log(1 + exp(input_data[offset + j] - 2 * input_data[offset + j] * (input_data[offset + j] >= 0)));
            }
        }
        Dtype count_all = count_pos + count_neg;
        if(count_all!=0) {
            loss_pos += temp_loss_pos * (1.0 * count_neg / count_all);
            loss_neg += temp_loss_neg * (1.0 * count_pos / count_all);
        } else {
            LOG(FATAL) << "All pixels ignored!";
        }
    }
    top[0]->mutable_cpu_data()[0] = (loss_pos * 1 + loss_neg) / num;
}

template <typename Dtype>
void ReweightedSigmoidCrossEntropyLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
    }
    if (propagate_down[0]) {
        // First, compute the diff
        const int count = bottom[0]->count();
        const int num = bottom[0]->num();
        const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();
        const Dtype* target = bottom[1]->cpu_data();
        const unsigned int ignored_bit = (1 << 31);
        const int count1 = bottom[1]->count();

        Dtype* pNewTarget = new Dtype[count1];
        for(int i=0; i<count1; ++i) {
            const unsigned int* ptr = reinterpret_cast<const unsigned int*>(target+i);
            const bool ignored = (*ptr) & ignored_bit;
            if(ignored) {
                pNewTarget[i] = -100;
                //LOG(FATAL) << "pixel ignored!"; //test added by cfeng
            } else {
                pNewTarget[i] = (int)((*ptr)!=0);
            }
        }
        Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
        caffe_sub(count, sigmoid_output_data, pNewTarget, bottom_diff);

        Dtype count_pos = 0;
        Dtype count_neg = 0;
        int dim = bottom[0]->count() / bottom[0]->num();

        for (int i = 0; i < num; ++i) {
        	count_pos = 0;
        	count_neg = 0;
        	const int offset = i*dim;
        	for (int j = 0; j < dim; j ++) {
                if (pNewTarget[offset+j] == 1) {
                    count_pos ++;
            	}
            	else if (pNewTarget[offset+j] == 0) {
                    count_neg ++;
            	}
         	}
         	const Dtype count_all = count_pos + count_neg;
         	Dtype weight_pos = 0;
         	Dtype weight_neg = 0;
         	if (count_all!=0) {
         	    weight_pos = 1.0 * count_neg / count_all;
         	    weight_neg = 1.0 * count_pos / count_all;
         	}
        	for (int j = 0; j < dim; j ++) {
                if (pNewTarget[offset+j] == 1) {
                    bottom_diff[offset+j] *= weight_pos;
            	}
            	else if (pNewTarget[offset+j] == 0) {
                    bottom_diff[offset+j] *= weight_neg;
            	} else {
            	    bottom_diff[offset+j] = 0;
            	}
         	}
        }
        delete[] pNewTarget;
        const Dtype loss_weight = top [0]->cpu_diff()[0];
        caffe_scal(count, loss_weight / num, bottom_diff);
    }
}

#ifdef CPU_ONLY
STUB_GPU_BACKWARD(ReweightedSigmoidCrossEntropyLossLayer, Backward);
#endif

INSTANTIATE_CLASS(ReweightedSigmoidCrossEntropyLossLayer);
REGISTER_LAYER_CLASS(ReweightedSigmoidCrossEntropyLoss);

}  // namespace caffe
