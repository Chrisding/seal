#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layers/domain_transform_layer.hpp"
#include "caffe/filler.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"
#include "caffe/util/io.hpp"

#include "gtest/gtest.h"

namespace caffe {

template <typename TypeParam>
class DomainTransformLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  DomainTransformLayerTest()
      : blob_top_(new Blob<Dtype>()),
        blob_bottom_input_(new Blob<Dtype>()),
        blob_bottom_gradient_(new Blob<Dtype>()),
        blob_bottom_input_dim_(new Blob<Dtype>()) {}

  void TestGradientSetUp() {
    // Set the random seed so that the generated inputs are reproducible.
    Caffe::set_random_seed(1);
    int num          = 3;
    int channel      = 2;
    int height       = 3;
    int width        = 4;
    int input_height = 2;
    int input_width  = 3;

    blob_bottom_input_->Reshape(num, channel, height, width);
    blob_bottom_gradient_->Reshape(num, 1, height, width);
    blob_bottom_input_dim_->Reshape(num, 2, 1, 1);

    for (int n = 0; n < num; ++n) {
      Dtype* input_dim_data = blob_bottom_input_dim_->mutable_cpu_data_at(n);
      // Change the input_height and input_width to complicate the test cases.
      switch (n) {
        case 0:
          input_dim_data[0] = input_height;
          input_dim_data[1] = input_width;
          break;
        case 1:
          input_dim_data[0] = input_height + 1;
          input_dim_data[1] = input_width  - 1;
          break;
        case 2:
          input_dim_data[0] = input_height + 1;
          input_dim_data[1] = input_width  + 1;
          break;
        default:
          input_dim_data[0] = input_height;
          input_dim_data[1] = input_width;
          break;
      }
    }

    FillerParameter filler_param;
    filler_param.set_min(0.01);
    GaussianFiller<Dtype> gauFiller(filler_param);
    UniformFiller<Dtype> uniFiller(filler_param);
    gauFiller.Fill(this->blob_bottom_input_);
    uniFiller.Fill(this->blob_bottom_gradient_);

    blob_bottom_vec_.push_back(blob_bottom_input_);
    blob_bottom_vec_.push_back(blob_bottom_gradient_);
    blob_bottom_vec_.push_back(blob_bottom_input_dim_);
    blob_top_vec_.push_back(blob_top_);
  }

  void TestForwardSetUp() {
    input_filename_ =
        "src/caffe/test/test_data/dt_input.txt";
    gradient_filename_ =
        "src/caffe/test/test_data/dt_ref_img_gradient.txt";
    expected_result_filename_ =
        "src/caffe/test/test_data/dt_output.txt";

    int row = 32;
    int col = 32;

    blob_top_->Reshape(1, 1, row, col);
    blob_bottom_input_->Reshape(1, 1, row, col);
    blob_bottom_gradient_->Reshape(1, 1, row, col);
    blob_bottom_input_dim_->Reshape(1, 1, 1, 2);

    Dtype* input_data = blob_bottom_input_->mutable_cpu_data();
    Dtype* grad_data  = blob_bottom_gradient_->mutable_cpu_data();

    CHECK(ReadDataFromFile(input_filename_, input_data))
        << "Fail to load " << input_filename_;
    CHECK(ReadDataFromFile(gradient_filename_, grad_data))
        << "Fail to load " << gradient_filename_;

    Dtype* dim_data = blob_bottom_input_dim_->mutable_cpu_data();
    dim_data[0] = row;
    dim_data[1] = col;

    blob_bottom_vec_.push_back(blob_bottom_input_);
    blob_bottom_vec_.push_back(blob_bottom_gradient_);
    blob_bottom_vec_.push_back(blob_bottom_input_dim_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~DomainTransformLayerTest() {
    delete blob_top_;
    delete blob_bottom_input_;
    delete blob_bottom_gradient_;
    delete blob_bottom_input_dim_;
  }

  bool ReadDataFromFile(const std::string& filename, Dtype* data) {
    std::ifstream ifs(filename.c_str(), std::ios_base::in);

    if (!ifs.is_open()) {
      return false;
    }

    std::string line;
    for (int count = 0; getline(ifs, line); ++count) {
      std::istringstream iss(line);
      iss >> data[count];
    }

    return true;
  }

  void TestForward(const Dtype kSigmaXY,
                   const Dtype kSigmaRGB,
                   const int kNumIter,
		   const Dtype kMinWeight) {
    LayerParameter layer_param;
    DomainTransformParameter* dt_param =
        layer_param.mutable_domain_transform_param();
    dt_param->set_spatial_sigma(kSigmaXY);
    dt_param->set_range_sigma(kSigmaRGB);
    dt_param->set_num_iter(kNumIter);
    dt_param->set_min_weight(kMinWeight);

    DomainTransformLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    int height      = blob_top_vec_[0]->height();
    int width       = blob_top_vec_[0]->width();
    int channels    = blob_top_vec_[0]->channels();
    int spatial_dim = height * width;

    const Dtype* result = blob_top_vec_[0]->cpu_data();

    int ind = 0;
    double kTolerance = 1e-6;
    Dtype* expected_result = new Dtype[spatial_dim * channels];

    // Read expected result.
    CHECK(ReadDataFromFile(expected_result_filename_,
                           expected_result))
        << "Fail to load " << expected_result_filename_;

    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        for (int c = 0; c < channels; ++c) {
          EXPECT_NEAR(result[ind], expected_result[ind], kTolerance);
          ++ind;
        }
      }
    }
  }

  string input_filename_;
  string gradient_filename_;
  string expected_result_filename_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_bottom_input_;
  Blob<Dtype>* const blob_bottom_gradient_;
  Blob<Dtype>* const blob_bottom_input_dim_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(DomainTransformLayerTest, TestDtypesAndDevices);

TYPED_TEST(DomainTransformLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  Dtype spatial_sigma = 8;
  Dtype range_sigma   = 120;
  Dtype num_iter      = 3;
  Dtype min_weight    = 0;
  this->TestForwardSetUp();
  this->TestForward(spatial_sigma, range_sigma, num_iter, min_weight);
}

TYPED_TEST(DomainTransformLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  Dtype spatial_sigma = 8;
  Dtype range_sigma   = 120;
  Dtype num_iter      = 3;
  LayerParameter layer_param;
  DomainTransformParameter* dt_param =
      layer_param.mutable_domain_transform_param();
  dt_param->set_spatial_sigma(spatial_sigma);
  dt_param->set_range_sigma(range_sigma);
  dt_param->set_num_iter(num_iter);

  DomainTransformLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);

  this->TestGradientSetUp();
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
           this->blob_top_vec_, 0);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
           this->blob_top_vec_, 1);
}

}  // namespace caffe
