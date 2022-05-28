//
// Created by 孙嘉禾 on 2019/12/31.
//

#include "deformable_conv2d.h"

#include <cmath>
#include <iostream>
#include <algorithm>
#include <string>

#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "deformable_conv2d_utils.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

REGISTER_OP("DeformableConv2D")
.Input("input: T")
.Input("filter: T")
.Input("offset: T")
.Input("mask: T")
.Output("output: T")
.Attr("T: {float, double}")
.Attr("strides: list(int)")
// .Attr("use_cudnn_on_gpu: bool = true")
.Attr("num_groups: int")
.Attr("deformable_groups: int")
.Attr("im2col_step: int")
.Attr("no_bias: bool = true")
.
Attr (GetPaddingAttrString())
.Attr("data_format: {'NCHW' } = 'NCHW' ")
.Attr("dilations: list(int) = [1, 1, 1, 1]")
.SetShapeFn([](
InferenceContext *c
) {
ShapeHandle input_shape;
TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input_shape));
ShapeHandle filter_shape;
TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 4, &filter_shape));
ShapeHandle offset_shape;
TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 4, &offset_shape));
ShapeHandle mask_shape;
TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 4, &mask_shape));

std::vector<int32> strides;
TF_RETURN_IF_ERROR(c->GetAttr("strides", &strides));
if (strides.
size()
!= 4) {
return errors::InvalidArgument(
"Deformconv requires the stride attribute to contain 4 values, but "
"got: ",
strides.
size()
);
}

std::vector<int32> rates;
TF_RETURN_IF_ERROR(c->GetAttr("dilations", &rates));
if (rates.
size()
!= 4) {
return errors::InvalidArgument(
"Deformconv requires the dilations attribute to contain 4 values, but "
"got: ",
rates.
size()
);
}
string data_format;
TensorFormat data_format_;
TF_RETURN_IF_ERROR(c->GetAttr("data_format", &data_format));
FormatFromString(data_format, &data_format_
);
const int32 stride_rows = GetTensorDim(strides, data_format_, 'H');
const int32 stride_cols = GetTensorDim(strides, data_format_, 'W');

const int32 rate_rows = GetTensorDim(rates, data_format_, 'H');
const int32 rate_cols = GetTensorDim(rates, data_format_, 'W');

int groups;
TF_RETURN_IF_ERROR(c->GetAttr("num_groups", &groups));
int deform_groups;
TF_RETURN_IF_ERROR(c->GetAttr("deformable_groups", &deform_groups));

DimensionHandle batch_size_dim = c->Dim(input_shape, 0);
DimensionHandle in_depths_dim = c->Dim(input_shape, 1);
DimensionHandle in_rows_dim = c->Dim(input_shape, 2);
DimensionHandle in_cols_dim = c->Dim(input_shape, 3);
DimensionHandle filter_rows_dim = c->Dim(filter_shape, 2);
DimensionHandle filter_cols_dim = c->Dim(filter_shape, 3);
DimensionHandle filter_depth_dim = c->Dim(filter_shape, 1);
DimensionHandle output_depth_dim = c->Dim(filter_shape, 0);
DimensionHandle multiplied_depth;
DimensionHandle depth_per_dfgps;
auto filter_row = c->Value(filter_rows_dim);
auto filter_col = c->Value(filter_cols_dim);
auto offset_dpt = c->Value(c->Dim(offset_shape, 1));
if ((offset_dpt % (
filter_row *filter_col
) != 0)
|| (offset_dpt / (2 *
filter_row *filter_col
) != deform_groups)) {
return errors::InvalidArgument(
"Deformconv requires the offset compatible with filter, but "
"got: ",
c->
DebugString(offset_shape)
);
}

auto mask_dpt = c->Value(c->Dim(mask_shape, 1));
if ((mask_dpt % (
filter_row *filter_col
) != 0) || (mask_dpt / (
filter_row *filter_col
) != deform_groups)) {
return errors::InvalidArgument("Deformconv requires the mask compatible with filter, but "
"got: ",
c->
DebugString(offset_shape)
);
}

TF_RETURN_IF_ERROR(
    c->Multiply(filter_depth_dim, groups, &multiplied_depth));
TF_RETURN_IF_ERROR(c->Divide(filter_depth_dim, deform_groups, true, &depth_per_dfgps));
TF_RETURN_IF_ERROR(c->Divide(in_depths_dim, deform_groups, true, &depth_per_dfgps));

if (!c->
ValueKnown(in_rows_dim)
|| !c->
ValueKnown(in_cols_dim)
||
!c->
ValueKnown(filter_rows_dim)
|| !c->
ValueKnown(filter_cols_dim)
) {
ShapeHandle output_shape =
    c->MakeShape({batch_size_dim, output_depth_dim, InferenceContext::kUnknownDim,
                  InferenceContext::kUnknownDim
                 });
c->set_output(0, output_shape);
return
Status::OK();
}
DimensionHandle unused;
TF_RETURN_IF_ERROR(
    c->Merge(c->Dim(input_shape, 1), multiplied_depth, &unused));

auto in_rows = c->Value(in_rows_dim);
auto in_cols = c->Value(in_cols_dim);
auto filter_rows = c->Value(filter_rows_dim);
auto filter_cols = c->Value(filter_cols_dim);
auto filter_rows_eff = filter_rows + (filter_rows - 1) * (rate_rows - 1);
auto filter_cols_eff = filter_cols + (filter_cols - 1) * (rate_cols - 1);

Padding padding;
TF_RETURN_IF_ERROR(c->GetAttr("padding", &padding));

int64 output_rows, output_cols;
int64 padding_before, padding_after;
TF_RETURN_IF_ERROR(GetWindowedOutputSizeVerbose(
    in_rows, filter_rows_eff, stride_rows, padding, &output_rows,
    &padding_before, &padding_after));
TF_RETURN_IF_ERROR(GetWindowedOutputSizeVerbose(
    in_cols, filter_cols_eff, stride_cols, padding, &output_cols,
    &padding_before, &padding_after));

ShapeHandle output_shape = c->MakeShape(
    {batch_size_dim, output_depth_dim, output_rows, output_cols});
c->set_output(0, output_shape);
// shape_inference::ShapeHandle offset_shape = c->input(2);
// shape_inference::ShapeHandle mask_shape = c->input(3);
shape_inference::DimensionHandle offset_batch = c->Dim(offset_shape, 0);
shape_inference::DimensionHandle offset_channel = c->Dim(offset_shape, 1);
shape_inference::DimensionHandle offset_height = c->Dim(offset_shape, 2);
shape_inference::DimensionHandle offset_weight = c->Dim(offset_shape, 3);
shape_inference::DimensionHandle mask_channel = c->Dim(mask_shape, 1);
shape_inference::DimensionHandle mask_height = c->Dim(mask_shape, 2);
shape_inference::DimensionHandle mask_weight = c->Dim(mask_shape, 3);
shape_inference::DimensionHandle mask_batch = c->Dim(mask_shape, 0);
TF_RETURN_IF_ERROR(c->WithRank(offset_shape, 4, &offset_shape));
TF_RETURN_IF_ERROR(c->WithRank(mask_shape, 4, &mask_shape));
TF_RETURN_IF_ERROR(c->WithValue(offset_batch, c->Value(batch_size_dim), &offset_batch));
TF_RETURN_IF_ERROR(c->WithValue(offset_channel,
                                2 * c->Value(filter_rows_dim) * c->Value(filter_cols_dim),
                                &offset_channel));
TF_RETURN_IF_ERROR(c->WithValue(offset_height, output_rows, &offset_height));
TF_RETURN_IF_ERROR(c->WithValue(offset_weight, output_cols, &offset_weight));
TF_RETURN_IF_ERROR(c->WithValue(mask_batch, c->Value(batch_size_dim), &mask_batch));
TF_RETURN_IF_ERROR(c->WithValue(mask_channel,
                                c->Value(filter_rows_dim) * c->Value(filter_cols_dim),
                                &mask_channel));
TF_RETURN_IF_ERROR(c->WithValue(mask_height, output_rows, &mask_height));
TF_RETURN_IF_ERROR(c->WithValue(mask_weight, output_cols, &mask_weight));
return
Status::OK();
})
.Doc(R"doc(
        DeformableConv2D is a new convolution operation with the deformable kernel locations.
        The inputs should have format NCHW, which is faster on GPUS.
        The offset and mask should have same input spatial resolution.
        Also, the output's shape depends on the stride, and I only consider the situation of dilation rate = 1.
    )doc");

// Opkernel defination.
// template parameter <T> is the datatype of the tensors
// in my opnion, the deformable convolution op ought to be implemented by extending the Conv2DOp, however, we can not get the conv_ops.h file if we choose to dynamic link the op

REGISTER_OP("DeformableConv2DBackProp")
.Input("input: T")
.Input("filter: T")
.Input("offset: T")
.Input("mask: T")
.Input("out_grad: T")
.Output("x_grad: T")
.Output("filter_grad: T")
.Output("offset_grad: T")
.Output("mask_grad: T")
.Attr("T: {float, double}")
.Attr("strides: list(int)")
// .Attr("use_cudnn_on_gpu: bool = true")
.Attr("num_groups: int")
.Attr("deformable_groups: int")
.Attr("im2col_step: int")
.Attr("no_bias: bool = true")
.
Attr (GetPaddingAttrString())
.Attr("data_format: { 'NCHW' } = 'NCHW' ")
.Attr("dilations: list(int) = [1, 1, 1, 1]")
.SetShapeFn([](
InferenceContext *c
) {
c->set_output(0, c->input(0));
c->set_output(1, c->input(1));
c->set_output(2, c->input(2));
c->set_output(3, c->input(3));
return
Status::OK();
})
.Doc(R"doc(only support NCHW now)doc");

REGISTER_OP("DeformablePsroiPool")
.Input("input: T")
.Input("bbox: T")
.Input("trans: T")
.Output("output: T")
.Output("top_count: T")
.Attr("T: {float, double}")
.Attr("pooled_size: int")
.Attr("no_trans: int")
.Attr("spatial_scale: float")
.Attr("output_dim: int")
.Attr("group_size: int")
.Attr("part_size: int")
.Attr("sample_per_part: int")
.Attr("trans_std: float")
.SetShapeFn([](
InferenceContext *ctx
) {
int pooled_size;
int output_dim;
TF_RETURN_IF_ERROR(ctx->GetAttr("pooled_size", &pooled_size));
TF_RETURN_IF_ERROR(ctx->GetAttr("output_dim", &output_dim));
auto input_handle = ctx->input(0);
auto batch = ctx->Dim(input_handle, 0);
auto output_dim_handle = ctx->MakeDim(output_dim);
auto pooled_size_handle = ctx->MakeDim(pooled_size);
ctx->set_output(0, ctx->MakeShape({
batch, output_dim_handle, pooled_size_handle, pooled_size_handle}));
ctx->set_output(1, ctx->MakeShape({
batch, output_dim_handle, pooled_size_handle, pooled_size_handle}));
return
Status::OK();
})
.Doc(R"doc(DeformablePsROIPool is a new pooling operation with the deformable
kernel locations. The inpus should have format NCHW, which is faster on GPUS.)doc");
REGISTER_OP("DeformablePsroiPoolBackProp")
.Input("data: T")
.Input("bbox: T")
.Input("trans: T")
.Input("top_count: T")
.Input("out_grad: T")
.Output("in_grad: T")
.Output("trans_grad: T")
.Attr("pooled_size: int")
.Attr("T: {float, double}")
.Attr("no_trans: int")
.Attr("spatial_scale: float")
.Attr("output_dim: int")
.Attr("group_size: int")
.Attr("part_size: int")
.Attr("sample_per_part: int")
.Attr("trans_std: float")
.SetShapeFn([](
InferenceContext *ctx
) {
ctx->set_output(0, ctx->input(0));
ctx->set_output(1, ctx->input(2));
return
Status::OK();
})
.Doc(R"doc("BackProp operation for DeformablePSROIPool")doc");

template<typename Device, typename T>
class DeformableConv2DOp : public OpKernel {
 public:
  explicit DeformableConv2DOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, InitDeformableConv2DParameters(ctx, &params_));
  }
  void Compute(OpKernelContext *context) override {
    // Input tensor's shape
    // [batch, channels, height, weight]
    const Tensor &input = context->input(0);
    const TensorShape &input_shape = input.shape();
    // [out_channels, in_channels, filter_height, filter_weight]
    const Tensor &filter = context->input(1);
    const TensorShape &filter_shape = filter.shape();
    // [batch, 2 * filter.Size(), out_height, out_weight]
    const Tensor &offset = context->input(2);
    const TensorShape &offset_shape = offset.shape();
    // [batch, filter.Size(), out_height, out_weight]
    const Tensor &mask = context->input(3);
    const TensorShape &mask_shape = mask.shape();

    DeformableConv2DDimensions dimensions;
    OP_REQUIRES_OK(context, ComputeDeformableConv2DDimension(params_, input, filter, &dimensions, 0));
    //data_format = NCHW
    // 这个地方我出了bug,原因是shapefromformat的参数必须是data_format, N, H, W, C,因为其内部是根据data_format来决定是否需要进行transpose, 如何第三个参数给了C, 且第一个参数为NCHW,那最后得到的结果会是NWCH
    TensorShape out_shape = ShapeFromFormat(
        params_.data_format, dimensions.batch, dimensions.out_rows,
        dimensions.out_cols, dimensions.out_depth);

    // Output tensor is of the following dimensions:
    // [ in_batch, out_depth, out_rows, out_cols]
    // Tensor* output = nullptr;
    // OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));
    VLOG(2) << "DeformableConv2D: in_depth = " << dimensions.in_depth
            << ", patch_depth = " << dimensions.patch_depth
            << ", input_cols = " << dimensions.input_cols
            << ", filter_cols = " << dimensions.filter_cols
            << ", input_rows = " << dimensions.input_rows
            << ", filter_rows = " << dimensions.filter_rows
            << ", stride_rows = " << dimensions.stride_rows
            << ", stride_cols = " << dimensions.stride_cols
            << ", dilation_rows = " << dimensions.dilation_rows
            << ", dilation_cols = " << dimensions.dilation_cols
            << ", out_depth = " << dimensions.out_depth;

    // If there is nothing to compute, return.
    if (out_shape.num_elements() == 0) {
      return;
    }

    /**
     * from here i stop use the traditional convolution implement of the official code which was defined in conv_ops.cc
     * and began to use the implement of the deformable conv2d of the msra version
     * **/
    LayerSetUp(input_shape, filter_shape, offset_shape, mask_shape, out_shape);
    // notice the fact that the flat function return a reference of a pointer, but in fact we only need a pointer
    const T *in_data_ptr = input.template flat<T>().data();
    const T *offset_ptr = offset.template flat<T>().data();
    const T *mask_ptr = mask.template flat<T>().data();
    const Device &d = context->eigen_device<Device>();
    int col_buffer_shape_temp[4];// calculate the shape of col_buffer, mxnet源码是 + 1, 多了一个im2col_step_
    col_buffer_shape_temp[0] =
        ProdShape(filter_shape, 1, filter_shape.dims());// 卷积核的参数个数,注意卷积核的形状应该是[out_depth, in_depth, height, weight]
    col_buffer_shape_temp[1] = im2col_step_;
    col_buffer_shape_temp[2] = out_shape.dim_size(2);
    col_buffer_shape_temp[3] = out_shape.dim_size(3);
    TensorShape col_buffer_shape = TensorShape({
                                                   col_buffer_shape_temp[0],
                                                   col_buffer_shape_temp[1],
                                                   col_buffer_shape_temp[2],
                                                   col_buffer_shape_temp[3]});

    Tensor col_buffer;
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value, col_buffer_shape, &col_buffer));
    T *col_buffer_ptr = col_buffer.template flat<T>().data();

    int32_t M = conv_out_channels_ / group_; // filter的数量
    int32_t N = im2col_step_ * conv_out_spatial_dim_;
    int32_t K = kernel_dim_; // 卷积的参数个数

    Tensor weight_3d;
    TensorShape weight_3d_shape = TensorShape({group_, M, K});
    OP_REQUIRES(context, weight_3d.CopyFrom(filter, weight_3d_shape), errors::InvalidArgument("shape doesn't match"));
    T *weight_3d_ptr = weight_3d.template flat<T>().data();

    Tensor *output_temp_4d = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output_temp_4d));
    auto output_temp_4d_ptr = output_temp_4d->template flat<T>().data();
//      auto output__ptr = output_temp_4d->flat<T>();
    /**
     * 这样的话下面计算矩阵乘法的时候直接就写到这个输出里了
     * 但是注意的是作者实现的时候划分ｓｔｅｐ，这个时候其实是往ｓｈａｐｅ为｛num_ / im2col_step_, group_, M, N｝的输出里写的，所以最后一定要置换一下维度的位置
     * **/
    SetZeros<Device, T>()(d, ProdShape(out_shape, 0, out_shape.dims()), output_temp_4d_ptr);
    TShape pads;
    pads.push_back(dimensions.pad_rows);
    pads.push_back(dimensions.pad_cols);
    for (int32_t n = 0; n < num_ / im2col_step_; ++n) { // 分batch进行
      // transform image to col_buffer in order to use gemm
      DeformableConv2DIm2Col<Device, T>()(
          d,
          in_data_ptr
              + n * im2col_step_ * input_dim_, // dptr是获取输入数据的指针 + n * im2col_step_* input_dim 是让指针向后移动 一张图片的数据
          offset_ptr + n * im2col_step_ * input_offset_dim_, //
          mask_ptr + n * im2col_step_ * input_mask_dim_,
          ToVector(input_shape),
          ToVector(col_buffer_shape),
          SubVector(filter_shape, 2, 4),
          pads,
          SubVector(params_.strides, 2, 4),
          SubVector(params_.dilations, 2, 4),
          params_.deformable_groups,
          col_buffer_ptr
      );
      TensorShape col_buffer_3d_shape = TensorShape({group_, K, N});

      auto output_temp_group_ptr = output_temp_4d_ptr + (n * group_ * M * N);

      LaunchBatchMatMul<Device, T>::launch(context,
                                           weight_3d_shape,
                                           col_buffer_3d_shape,
                                           weight_3d_ptr,
                                           col_buffer_ptr,
                                           false,
                                           false,
                                           output_temp_group_ptr);

    }
  }
 private:
  DeformableConv2DParameters params_;
  bool use_cudnn_;
  bool cudnn_use_autotune_;
  int32_t channel_axis_;  // channel axis of the input
  int32_t channels_;  // number of channels of input image
  int32_t num_spatial_axes_;  // number of spatial axes
  int32_t num_;  // batch size
  int32_t group_;  // number of groups
  int32_t conv_out_channels_;  // number of output channels (num_filter)
  int32_t conv_out_spatial_dim_;  // number of pixels of output images per channel
  int32_t conv_in_channels_;  // number of input channels
  int32_t kernel_dim_;  // number of input channels per group * kernel size
  int32_t weight_offset_;  // number of output channels per group * kernel_dim_
  int32_t col_offset_;
  int32_t output_offset_;
  int32_t col_buffer_size_;
  int32_t input_dim_;
  int32_t input_offset_dim_;
  int32_t input_mask_dim_;
  int32_t output_dim_;
  int32_t num_kernels_im2col_;
  int32_t num_kernels_col2im_;
  int32_t im2col_step_;
  bool bias_term_;  // has bias term?
  bool is_1x1_;
  void LayerSetUp(const TensorShape &ishape,
                  const TensorShape &filter_shape,
                  const TensorShape &offset_shape,
                  const TensorShape &mask_shape,
                  const TensorShape &oshape) {
    channel_axis_ = 1;  // hard code channel axis, fixed the input data_format
    const int32_t first_spatial_axis = channel_axis_ + 1;
    const int32_t num_axes = filter_shape.dims();
    num_spatial_axes_ = num_axes - first_spatial_axis; //表示的是空间坐标个数,比如说2维卷积里,就是2, 3维卷积里就是3
    is_1x1_ = true; //  判断是否为1x1卷积
    for (int32_t i = 2; i < filter_shape.dims(); ++i) {
      // is_1x1_ &= filter_shape.dim_size(i) == 1 && params_.stride[i] == 1 && params_.pad[i] == 0;
      is_1x1_ &= filter_shape.dim_size(i) == 1; // only judge by the filter's shape
      if (!is_1x1_) break;
    }
    num_ = ishape.dim_size(0);// batch size
    channels_ = ishape.dim_size(1);// number of input channels
    group_ = params_.num_groups;//
    conv_out_channels_ = filter_shape.dim_size(0); // output channel nums
    conv_in_channels_ = channels_; // input channel nums
    bias_term_ = !params_.no_bias; //
    kernel_dim_ = conv_in_channels_ / group_ * filter_shape.dim_size(2)
        * filter_shape.dim_size(3); //Size()返回tensor中元素个数，即各维度大小的乘积，所以这里的kernel_dim的意思是卷积核的参数个数了．
    conv_out_spatial_dim_ = ProdShape(oshape,
                                      2,
                                      oshape.dims()); //ProdShape(dimstart, dimend)返回指定维度大小乘积, 这个变量代表每个通道的像素点个数, oshape.ndim()返回这个shape的维度，假设是NCHW那么返回4,则为 H * W，
//        col_offset_ = kernel_dim_ * conv_out_spatial_dim_;//kernel_dim代表一个卷积核参数的个数，conv_out_spatial_dim_相当于特征图上的坐标个数，那这个变量相当于总共需要的偏移量
//        weight_offset_ = conv_out_channels_ * kernel_dim_ / group_;//这里应该是所有的权重的个数，也就是需要求的权重偏移的个数
//        output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ / group_;//这里是输出通道数乘上每个通道的像素点的个数，所以结果应该是输出的总维度，就是C*H*W
    im2col_step_ = std::min(params_.im2col_step, num_);
    col_buffer_size_ = kernel_dim_ * group_ * im2col_step_
        * conv_out_spatial_dim_;// 开辟的缓存大小// size of the column buffer used for storing im2col-ed pixels

    input_dim_ = ProdShape(ishape, 1, ishape.dims());// input image size (#channels * height * width)
    input_offset_dim_ = ProdShape(offset_shape, 1, offset_shape.dims()); // 18 * H * W
    input_mask_dim_ = ProdShape(mask_shape, 1, mask_shape.dims()); // 9 * H * W
    output_dim_ = ProdShape(oshape, 1, oshape.dims()); //输出的元素个数

    num_kernels_im2col_ =
        conv_in_channels_ * conv_out_spatial_dim_; //如果输出和输入的分辨率不变的话，代表输入数据的dim,我个人觉得就是把整个输入展开为一个一维向量,在求其维度大小
    num_kernels_col2im_ = input_dim_;//输入数据的dim
  }
};

template<typename Device, typename T>
class DeformableConv2DBackPropOp : public OpKernel {
 public:
  explicit DeformableConv2DBackPropOp(OpKernelConstruction *context) : OpKernel(context) {
    OP_REQUIRES_OK(context, InitDeformableConv2DParameters(context, &params_));
  }
  void Compute(OpKernelContext *ctx) override {
    const Tensor &x = ctx->input(0);
    const TensorShape &x_shape = x.shape();
    const Tensor &filter = ctx->input(1);
    const TensorShape &filter_shape = filter.shape();
    const Tensor &offset = ctx->input(2);
    const TensorShape &offset_shape = offset.shape();
    const Tensor &mask = ctx->input(3);
    const TensorShape &mask_shape = mask.shape();
    const Tensor &out_grad = ctx->input(4);
    const TensorShape &out_grad_shape = out_grad.shape();
    DeformableConv2DDimensions dimensions;
    OP_REQUIRES_OK(ctx, ComputeDeformableConv2DDimension(params_, x, filter, &dimensions, 1));
    LayerSetUp(x_shape, filter_shape, offset_shape, mask_shape, out_grad_shape);
    const Device &d = ctx->eigen_device<Device>();
    int col_buffer_shape_temp[4];
    col_buffer_shape_temp[0] = ProdShape(filter_shape, 1, filter_shape.dims());
    col_buffer_shape_temp[1] = im2col_step_;
    col_buffer_shape_temp[2] = out_grad_shape.dim_size(2);
    col_buffer_shape_temp[3] = out_grad_shape.dim_size(3);
    TensorShape col_buffer_shape = TensorShape({
                                                   col_buffer_shape_temp[0],
                                                   col_buffer_shape_temp[1],
                                                   col_buffer_shape_temp[2],
                                                   col_buffer_shape_temp[3]
                                               });
    int32_t M = kernel_dim_;
    int32_t N = im2col_step_ * conv_out_spatial_dim_;
    int32_t K = conv_out_channels_ / group_;
    const auto x_ptr = x.template flat<T>().data();
    const auto offset_ptr = offset.template flat<T>().data();
    const auto mask_ptr = mask.template flat<T>().data();
    const auto weight_3d_ptr = filter.template flat<T>().data();
    TensorShape weight_3d_shape = TensorShape({group_, K, M});
    Tensor out_grad_4d;
    TensorShape out_grad_4d_shape =
        TensorShape({num_ / im2col_step_, im2col_step_, conv_out_channels_, conv_out_spatial_dim_});
    OP_REQUIRES(ctx,
                out_grad_4d.CopyFrom(out_grad, out_grad_4d_shape),
                errors::InvalidArgument("shape doesn't match"));
    auto out_grad_4d_ptr = out_grad_4d.template flat<T>().data();
    out_grad_4d_shape = TensorShape({num_ / im2col_step_, group_, K, N});
    Tensor col_buffer;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value, col_buffer_shape, &col_buffer));
    auto col_buffer_3d_ptr = col_buffer.template flat<T>().data();
    TensorShape col_buffer_3d_shape = TensorShape({group_, M, N});
    Tensor *dweight_3d = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, filter_shape, &dweight_3d));
    T *dweight_3d_ptr = dweight_3d->template flat<T>().data();
    Tensor *x_grad = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x_shape, &x_grad));
    T *x_grad_ptr = x_grad->template flat<T>().data();
    Tensor *offset_grad = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, offset_shape, &offset_grad));
    T *offset_grad_ptr = offset_grad->template flat<T>().data();

    Tensor *mask_grad = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(3, mask_shape, &mask_grad));
    T *mask_grad_ptr = mask_grad->template flat<T>().data();
    TShape pads;
    pads.push_back(dimensions.pad_rows);
    pads.push_back(dimensions.pad_cols);
    TShape kernel_shape = SubVector(filter_shape, 2, 4);
    TShape stride_shape = SubVector(params_.strides, 2, 4);
    TShape dilation_shape = SubVector(params_.dilations, 2, 4);
    Tensor dweight_3d_temp;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value, filter_shape, &dweight_3d_temp));
    T *dweight_3d_temp_ptr = dweight_3d_temp.template flat<T>().data();
    SetZeros<Device, T>()(d, group_ * M * N, col_buffer_3d_ptr);
    SetZeros<Device, T>()(d, ProdShape(x_shape, 0, x_shape.dims()), x_grad_ptr);
    SetZeros<Device, T>()(d, ProdShape(filter_shape, 0, filter_shape.dims()), dweight_3d_ptr);
    SetZeros<Device, T>()(d, ProdShape(filter_shape, 0, filter_shape.dims()), dweight_3d_temp_ptr);
    for (int n = 0; n < num_ / im2col_step_; ++n) {
      TensorShape out_grad_3d_shape = TensorShape({group_, K, N});
      T *out_grad_3d_ptr = out_grad_4d_ptr + n * group_ * K * N;
      LaunchBatchMatMul<Device, T>::launch(
          ctx, weight_3d_shape,
          out_grad_3d_shape,
          weight_3d_ptr,
          out_grad_3d_ptr,
          true, false,
          col_buffer_3d_ptr);
      DeformableConv2DCol2ImCoord<Device, T>()(d,
                                               col_buffer_3d_ptr,
                                               x_ptr + n * im2col_step_ * input_dim_,
                                               offset_ptr + n * im2col_step_ * input_offset_dim_,
                                               mask_ptr + n * im2col_step_ * input_mask_dim_,
                                               ToVector(x_shape), ToVector(col_buffer_shape),
                                               kernel_shape,
                                               pads,
                                               stride_shape,
                                               dilation_shape,
                                               params_.deformable_groups,
                                               offset_grad_ptr + n * im2col_step_ * input_offset_dim_,
                                               mask_grad_ptr + n * im2col_step_ * input_mask_dim_);
      DeformableConv2DCol2Im<Device, T>()(d,
                                          col_buffer_3d_ptr,
                                          offset_ptr + n * im2col_step_ * input_offset_dim_,
                                          mask_ptr + n * im2col_step_ * input_mask_dim_,
                                          ToVector(x_shape), ToVector(col_buffer_shape),
                                          kernel_shape,
                                          pads,
                                          stride_shape,
                                          dilation_shape,
                                          params_.deformable_groups,
                                          x_grad_ptr + n * im2col_step_ * input_dim_);
      DeformableConv2DIm2Col<Device, T>()(d,
                                          x_ptr + n * im2col_step_ * input_dim_,
                                          offset_ptr + n * im2col_step_ * input_offset_dim_,
                                          mask_ptr + n * im2col_step_ * input_mask_dim_,
                                          ToVector(x_shape), ToVector(col_buffer_shape),
                                          kernel_shape,
                                          pads,
                                          stride_shape,
                                          dilation_shape,
                                          params_.deformable_groups,
                                          col_buffer_3d_ptr);
      if (n == 0) {
        LaunchBatchMatMul<Device, T>::launch(
            ctx,
            out_grad_3d_shape,
            col_buffer_3d_shape,
            out_grad_3d_ptr,
            col_buffer_3d_ptr,
            false, true,
            dweight_3d_ptr);
      } else {
        LaunchBatchMatMul<Device, T>::launch(
            ctx,
            out_grad_3d_shape,
            col_buffer_3d_shape,
            out_grad_3d_ptr,
            col_buffer_3d_ptr,
            false, true,
            dweight_3d_temp_ptr);
        PureAddTo<Device, T>()(d,
                               ProdShape(filter_shape, 0, filter_shape.dims()),
                               dweight_3d_ptr,
                               dweight_3d_temp_ptr);
      }
    }
  }
 private:
  DeformableConv2DParameters params_;
  // bool use_cudnn_;
  // bool cudnn_use_autotune_;
  int32_t channel_axis_;  // channel axis of the input
  int32_t channels_;  // number of channels of input image
  int32_t num_spatial_axes_;  // number of spatial axes
  int32_t num_;  // batch size
  int32_t group_;  // number of groups
  int32_t conv_out_channels_;  // number of output channels (num_filter)
  int32_t conv_out_spatial_dim_;  // number of pixels of output images per channel
  int32_t conv_in_channels_;  // number of input channels
  int32_t kernel_dim_;  // number of input channels per group * kernel size
  int32_t weight_offset_;  // number of output channels per group * kernel_dim_
  int32_t col_offset_;
  int32_t output_offset_;
  int32_t col_buffer_size_;
  int32_t input_dim_;
  int32_t input_offset_dim_;
  int32_t input_mask_dim_;
  int32_t output_dim_;
  int32_t num_kernels_im2col_;
  int32_t num_kernels_col2im_;
  int32_t im2col_step_;
  bool bias_term_;  // has bias term?
  bool is_1x1_;
  void LayerSetUp(const TensorShape &ishape,
                  const TensorShape &filter_shape,
                  const TensorShape &offset_shape,
                  const TensorShape &mask_shape,
                  const TensorShape &oshape) {
    channel_axis_ = 1;  // hard code channel axis, fixed the input data_format
    const int32_t first_spatial_axis = channel_axis_ + 1;
    const int32_t num_axes = filter_shape.dims();
    num_spatial_axes_ = num_axes - first_spatial_axis; //表示的是空间坐标个数,比如说2维卷积里,就是2, 3维卷积里就是3
    is_1x1_ = true; //  判断是否为1x1卷积
    for (int32_t i = 2; i < filter_shape.dims(); ++i) {
      // is_1x1_ &= filter_shape.dim_size(i) == 1 && params_.stride[i] == 1 && params_.pad[i] == 0;
      is_1x1_ &= filter_shape.dim_size(i) == 1; // only judge by the filter's shape
      if (!is_1x1_) break;
    }
    num_ = ishape.dim_size(0);// batch size
    channels_ = ishape.dim_size(1);// number of input channels
    group_ = params_.num_groups;//
    conv_out_channels_ = filter_shape.dim_size(0); // output channel nums
    conv_in_channels_ = channels_; // input channel nums
    bias_term_ = !params_.no_bias; //
    kernel_dim_ = conv_in_channels_ / group_ * filter_shape.dim_size(2)
        * filter_shape.dim_size(3); //Size()返回tensor中元素个数，即各维度大小的乘积，所以这里的kernel_dim的意思是卷积核的参数个数了．
    conv_out_spatial_dim_ = ProdShape(oshape,
                                      2,
                                      oshape.dims()); //ProdShape(dimstart, dimend)返回指定维度大小乘积, 这个变量代表每个通道的像素点个数, oshape.ndim()返回这个shape的维度，假设是NCHW那么返回4,则为 H * W，
    col_offset_ = kernel_dim_
        * conv_out_spatial_dim_;//kernel_dim代表一个卷积核参数的个数，conv_out_spatial_dim_相当于特征图上的坐标个数，那这个变量相当于总共需要的偏移量
    weight_offset_ = conv_out_channels_ * kernel_dim_ / group_;//这里应该是所有的权重的个数，也就是需要求的权重偏移的个数
    output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ / group_;//这里是输出通道数乘上每个通道的像素点的个数，所以结果应该是输出的总维度，就是C*H*W
    im2col_step_ = std::min(params_.im2col_step, num_);
    col_buffer_size_ = kernel_dim_ * group_ * im2col_step_
        * conv_out_spatial_dim_;// 开辟的缓存大小// size of the column buffer used for storing im2col-ed pixels

    input_dim_ = ProdShape(ishape, 1, ishape.dims());// input image size (#channels * height * width)
    input_offset_dim_ = ProdShape(offset_shape, 1, offset_shape.dims()); // 18 * H * W
    input_mask_dim_ = ProdShape(mask_shape, 1, mask_shape.dims()); // 9 * H * W
    output_dim_ = ProdShape(oshape, 1, oshape.dims()); //输出的元素个数

    num_kernels_im2col_ =
        conv_in_channels_ * conv_out_spatial_dim_; //如果输出和输入的分辨率不变的话，代表输入数据的dim,我个人觉得就是把整个输入展开为一个一维向量,在求其维度大小
    num_kernels_col2im_ = input_dim_;//输入数据的dim
  };
};

template<typename Device, typename Type>
class DeformablePSROIPoolOp : public OpKernel {
 public:
  explicit DeformablePSROIPoolOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("pooled_size", &pool_size));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("no_trans", &no_trans));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("spatial_scale", &spatial_scale));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_dim", &output_dim));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("group_size", &group_size));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("part_size", &part_size));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("sample_per_part", &sample_per_part));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("trans_std", &trans_std));
  }
  void Compute(OpKernelContext *ctx) override {
    const Tensor &data = ctx->input(0);
    const Tensor &bbox = ctx->input(1);
    const Tensor &trans = ctx->input(2);
    const int batch = data.dim_size(0);
    const int channels = data.dim_size(1);
    const int height = data.dim_size(2);
    const int width = data.dim_size(3);
    const int channels_trans = no_trans ? 2 : trans.dim_size(1);
    const int num_bbox = bbox.dim_size(0);
    Tensor *output;
    Tensor *top_count;
    const int pooled_width = pool_size;
    const int pooled_height = pool_size;
    const int count = num_bbox * output_dim * pooled_height * pooled_width;
    const int num_classes = no_trans ? 1 : channels_trans / 2;
    const int channels_each_class = no_trans ? output_dim : output_dim / num_classes;
    TensorShape output_shape{num_bbox, output_dim, pooled_height, pooled_width};
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, output_shape, &top_count));
    const Type *bottom_data = data.flat<Type>().data();
    const Type *bottom_rois = bbox.flat<Type>().data();
    const Type *bottom_trans = no_trans ? nullptr : trans.flat<Type>().data();
    Type *top_data = output->flat<Type>().data();
    Type *top_count_data = top_count->flat<Type>().data();
    const Device &d = ctx->eigen_device<Device>();
    DeformablePSROIPoolForward<Device, Type>()(d, count, bottom_data, spatial_scale,
                                               channels, height, width, pooled_height, pooled_width, bottom_rois,
                                               bottom_trans, no_trans, trans_std, sample_per_part, output_dim,
                                               group_size, part_size, num_classes, channels_each_class,
                                               top_data, top_count_data);
  }
 private:
  int pool_size;
  int no_trans;
  float spatial_scale;
  int output_dim;
  int group_size;
  int part_size;
  int sample_per_part;
  float trans_std;
};

template<typename Device, typename Type>
class DeformablePSROIPoolBackPropOp : public OpKernel {
 public:
  explicit DeformablePSROIPoolBackPropOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("pooled_size", &pool_size));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("no_trans", &no_trans));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("spatial_scale", &spatial_scale));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_dim", &output_dim));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("group_size", &group_size));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("part_size", &part_size));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("sample_per_part", &sample_per_part));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("trans_std", &trans_std));
  }
  void Compute(OpKernelContext *ctx) override {
    const Tensor &data = ctx->input(0);
    const Tensor &bbox = ctx->input(1);
    const Tensor &trans = ctx->input(2);
    const Tensor &top_count = ctx->input(3);
    const Tensor &out_grad = ctx->input(4);
    const int batch = data.dim_size(0);
    const int channels = data.dim_size(1);
    const int height = data.dim_size(2);
    const int width = data.dim_size(3);
    const int channels_trans = no_trans ? 2 : trans.dim_size(1);
    const int num_bbox = bbox.dim_size(0);
    const int num_rois = num_bbox;
    const int pooled_height = pool_size;
    const int pooled_width = pool_size;
    const int count = num_bbox * output_dim * pooled_height * pooled_width;
    const int num_classes = no_trans ? 1 : channels_trans / 2;
    const int channels_each_class = no_trans ? output_dim : output_dim / num_classes;
    Tensor *in_grad = nullptr;
    Tensor *trans_grad = nullptr;
    const TensorShape &in_grad_shape = data.shape();
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, in_grad_shape, &in_grad));
    TensorShape trans_grad_shape;
    const Type *top_diff = out_grad.flat<Type>().data();
    const Type *bottom_data = data.flat<Type>().data();
    const Type *bottom_rois = bbox.flat<Type>().data();
    trans_grad_shape = trans.shape();
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, trans_grad_shape, &trans_grad));
    const Type *bottom_trans = no_trans ? nullptr : trans.flat<Type>().data();
    Type *bottom_data_diff = in_grad->flat<Type>().data();
    Type *bottom_trans_diff = no_trans ? nullptr : trans_grad->flat<Type>().data();
    const Type *top_count_data = top_count.flat<Type>().data();
    const Device &d = ctx->eigen_device<Device>();
    DeformablePSROIPoolBackwardKernel<Device, Type>()(d,
                                                      count,
                                                      top_diff,
                                                      top_count_data,
                                                      num_rois,
                                                      spatial_scale,
                                                      channels,
                                                      height,
                                                      width,
                                                      pooled_height,
                                                      pooled_width,
                                                      output_dim,
                                                      bottom_data_diff,
                                                      bottom_trans_diff,
                                                      bottom_data,
                                                      bottom_rois,
                                                      bottom_trans,
                                                      no_trans,
                                                      trans_std,
                                                      sample_per_part,
                                                      group_size,
                                                      part_size,
                                                      num_classes,
                                                      channels_each_class);
  }
 private:
  int pool_size;
  int no_trans;
  float spatial_scale;
  int output_dim;
  int group_size;
  int part_size;
  int sample_per_part;
  float trans_std;
};

#define REGISTER_CPU(T)             \
    REGISTER_KERNEL_BUILDER(        \
        Name("DeformableConv2D").Device(DEVICE_CPU).TypeConstraint<T>("T"),\
        DeformableConv2DOp<CPUDevice, T>);  \
        REGISTER_KERNEL_BUILDER(Name("DeformableConv2DBackProp").Device(DEVICE_CPU).TypeConstraint<T>("T"), DeformableConv2DBackPropOp<CPUDevice, T>);
REGISTER_CPU(float);
REGISTER_CPU(double);
#undef REGISTER_CPU
#define REGISTER_CPU(T)             \
    REGISTER_KERNEL_BUILDER(        \
        Name("DeformablePsroiPool").Device(DEVICE_CPU).TypeConstraint<T>("T"),\
        DeformablePSROIPoolOp<CPUDevice, T>); \
        REGISTER_KERNEL_BUILDER(Name("DeformablePsroiPoolBackProp").Device(DEVICE_CPU).TypeConstraint<T>("T"), DeformablePSROIPoolBackPropOp<CPUDevice, T>);
REGISTER_CPU(float);
REGISTER_CPU(double);
#undef REGISTER_CPU

#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)              \
    REGISTER_KERNEL_BUILDER(Name("DeformableConv2D").Device(DEVICE_GPU).TypeConstraint<T>("T"), DeformableConv2DOp<GPUDevice, T>); \
    REGISTER_KERNEL_BUILDER(Name("DeformableConv2DBackProp").Device(DEVICE_GPU).TypeConstraint<T>("T"), DeformableConv2DBackPropOp<GPUDevice, T>);
REGISTER_GPU(float);
REGISTER_GPU(double);
#undef REGISTER_GPU
#define REGISTER_GPU(T)              \
    REGISTER_KERNEL_BUILDER(Name("DeformablePsroiPool").Device(DEVICE_GPU).TypeConstraint<T>("T"), DeformablePSROIPoolOp<GPUDevice, T>); \
    REGISTER_KERNEL_BUILDER(Name("DeformablePsroiPoolBackProp").Device(DEVICE_GPU).TypeConstraint<T>("T"), DeformablePSROIPoolBackPropOp<GPUDevice, T>);
REGISTER_GPU(float);
REGISTER_GPU(double);
#endif

}
