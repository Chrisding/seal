## SEAL Caffe Distribution

### Introduction

This Caffe distribution is a modified version of the [Deeplab v2](https://bitbucket.org/aquariusjay/deeplab-public-ver2) Caffe distribution. Modifications are made to accommodate some new modules and features required by SEAL. The distribution also fully supports the experiments of [CASENet](https://arxiv.org/abs/1705.09759).

### Change Summary

We have made the following changes from the original Deeplab v2 caffe distribution:
1. Added a multichannel reweighted sigmoid cross-entropy loss layer for CASENet.
2. Added a multichannel sigmoid cross-entropy loss layer for SEAL and some baselines (CASENet-S/CASENet-C) in the paper.
3. Modified image seg data layer and data transformer accordingly to incorporate the ability to read the uint32 format .bin files containing the bitwise multi-label edge ground truths. (Not required by SEAL, but would support the PyCaffe CASENet and the future PyCaffe SEAL)
4. Modified the solver with improved MatCaffe APIs for better top-level solver control.

### Usage requirement

1. [matio](https://sourceforge.net/projects/matio/files/matio/1.5.2/) is required to successfully compile this distribution. Please refer to [Deeplab v2](https://bitbucket.org/aquariusjay/deeplab-public-ver2) for more details.
2. MatCaffe of this distribution only works for certain Matlab versions. The code has been tested successfully on Matlab R2014a under Ubuntu 14.04LTS.
