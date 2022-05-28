//
// Created by 孙嘉禾 on 2020/2/1.
//

#include "deformable_conv2d.h"
#include <atomic>
#include <algorithm>

namespace tensorflow {

using ull = unsigned long long int;
using uInt = unsigned int;
typedef Eigen::GpuDevice GPUDevice;
typedef Eigen::ThreadPoolDevice CPUDevice;

Eigen::IndexPair<Eigen::DenseIndex> ContractionDims(bool adj_x, bool adj_y) {
  return {adj_x ? 0 : 1, adj_y ? 1 : 0};
}

void AtomicAdd(float *address, float val) {
  auto *address_as_ull = reinterpret_cast<uInt*>(address);
  uInt old = *address_as_ull;
  uInt assumed;
  float desired;
  do {
    assumed = old;
    desired = *reinterpret_cast<float *>(&assumed) + static_cast<float>(val);
    old = __sync_val_compare_and_swap(address_as_ull, assumed, *reinterpret_cast<uInt*>(&desired));
  } while (assumed != old);
}

void AtomicAdd(double *address, double val) {
  auto *address_as_ull = reinterpret_cast<ull*>(address);
  ull old = *address_as_ull;
  ull assumed;
  double desired;
  do {
    assumed = old;
    desired = *reinterpret_cast<double *>(&assumed) + static_cast<double>(val);
    old = __sync_val_compare_and_swap(address_as_ull, assumed, *reinterpret_cast<ull *>(&desired));
  } while (assumed != old);
}


template<typename DType>
void SwapAxisKernel(const CPUDevice &d, const int n, const int cuda_mem_size, const int min_unit_size,
                    DType *input_data, const int dim_num, const int axis_x_dims, const int axis_y_dims,
                    const int axis_x, const int axis_y) {
  d.parallelFor(n,
                Eigen::TensorOpCost(cuda_mem_size, cuda_mem_size, cuda_mem_size * axis_y_dims * axis_x_dims),
                [min_unit_size, input_data, dim_num, axis_x_dims, axis_y_dims,
                    axis_x, axis_y, cuda_mem_size](int64 start, int64 end) {
                  for (int64 index = start; index < end; index++) {
                    auto *device_data = new DType[cuda_mem_size];
                    DType *input_data_ptr = input_data + index * cuda_mem_size;
                    for (int j = 0; j < axis_y_dims; j++) {
                      for (int i = 0; i < axis_x_dims; i++) {
                        DType *temp_ptr = input_data_ptr + (i * axis_x_dims + j) * min_unit_size;
                        DType *device_data_temp_ptr = device_data + (j * axis_y_dims + i) * min_unit_size;
                        for (int k = 0; k < min_unit_size; k++) {
                          *(device_data_temp_ptr + k) = *(temp_ptr + k);
                        }
                      }
                    }
                    for (int idx = 0; idx < cuda_mem_size; idx++) {
                      *(input_data_ptr + idx) = *(device_data + idx);
                    }
                    delete[] device_data;
                  }
                });
}

template<typename T>
void DeformablePSROIPoolBackwardCpuAccKernel(const CPUDevice &d,
                                             const int count,
                                             const T *top_diff,
                                             const T *top_count,
                                             const int num_rois,
                                             const T spatial_scale,
                                             const int channels,
                                             const int height,
                                             const int width,
                                             const int pooled_height,
                                             const int pooled_width,
                                             const int output_dim,
                                             T *bottom_data_diff,
                                             T *bottom_trans_diff,
                                             const T *bottom_data,
                                             const T *bottom_rois,
                                             const T *bottom_trans, const int no_trans,
                                             const T trans_std, const int sample_per_part,
                                             const int group_size, const int part_size,
                                             const int num_classes,
                                             const int channels_each_class) {
  auto f = [count, top_diff, top_count, num_rois, spatial_scale, channels, height,
      width, pooled_height, pooled_width, output_dim, bottom_data_diff,
      bottom_trans_diff, bottom_data, bottom_rois, bottom_trans,
      no_trans, trans_std, sample_per_part, group_size, part_size,
      num_classes, channels_each_class](int64 start, int64 end) {
    for (int64 index = start; index < end; ++index) {
      int pw = index % pooled_width;
      int ph = (index / pooled_width) % pooled_height;
      int ctop = (index / pooled_width / pooled_height) % output_dim;
      int n = index / pooled_width / pooled_height / output_dim;
      // [start, end) interval for spatial sampling
      const T *offset_bottom_rois = bottom_rois + n * 5;
      int roi_batch_ind = offset_bottom_rois[0];
      T roi_start_w = (T) (round(offset_bottom_rois[1])) * spatial_scale - 0.5;
      T roi_start_h = (T) (round(offset_bottom_rois[2])) * spatial_scale - 0.5;
      T roi_end_w = (T) (round(offset_bottom_rois[3]) + 1.) * spatial_scale - 0.5;
      T roi_end_h = (T) (round(offset_bottom_rois[4]) + 1.) * spatial_scale - 0.5;
      // Force too small ROIs to be 1x1
      T roi_width = std::max(roi_end_w - roi_start_w, static_cast<T>(0.1)); //avoid 0
      T roi_height = std::max(roi_end_h - roi_start_h, static_cast<T>(0.1));

      // Compute w and h at bottom
      T bin_size_h = roi_height / static_cast<T>(pooled_height);
      T bin_size_w = roi_width / static_cast<T>(pooled_width);

      T sub_bin_size_h = bin_size_h / static_cast<T>(sample_per_part);
      T sub_bin_size_w = bin_size_w / static_cast<T>(sample_per_part);

      int part_h = floor((T) (ph) / pooled_height * part_size);
      int part_w = floor((T) (pw) / pooled_width * part_size);
      int class_id = ctop / channels_each_class;
      T trans_x = no_trans ? static_cast<T>(0) :
                  bottom_trans[(((n * num_classes + class_id) * 2) * part_size + part_h) * part_size + part_w]
                      * (T) trans_std;
      T trans_y = no_trans ? (T) (0) :
                  bottom_trans[(((n * num_classes + class_id) * 2 + 1) * part_size + part_h) * part_size + part_w]
                      * (T) trans_std;

      T wstart = (T) (pw) * bin_size_w + roi_start_w;
      wstart += trans_x * roi_width;
      T hstart = (T) (ph) * bin_size_h + roi_start_h;
      hstart += trans_y * roi_height;

      if (top_count[index] <= 0) {
        continue;
      }
      T diff_val = top_diff[index] / top_count[index];
      const T *offset_bottom_data = bottom_data + roi_batch_ind * channels * height * width;
      T *offset_bottom_data_diff = bottom_data_diff + roi_batch_ind * channels * height * width;
      int gw = floor((T) (pw) * group_size / pooled_width);
      int gh = floor((T) (ph) * group_size / pooled_height);
      gw = std::min(std::max(gw, 0), group_size - 1);
      gh = std::min(std::max(gh, 0), group_size - 1);
      for (int ih = 0; ih < sample_per_part; ih++) {
        for (int iw = 0; iw < sample_per_part; iw++) {
          T w = wstart + iw * sub_bin_size_w;
          T h = hstart + ih * sub_bin_size_h;
          // bilinear interpolation
          if (w < -0.5 || w > width - 0.5 || h < -0.5 || h > height - 0.5) {
            continue;
          }
          w = std::min(std::max(w, static_cast<T>(0.)), static_cast<T>(width - 1.));
          h = std::min(std::max(h, static_cast<T>(0.)), static_cast<T>(height - 1.));
          int c = (ctop * group_size + gh) * group_size + gw;
          // backward on feature
          int x0 = floor(w);
          int x1 = ceil(w);
          int y0 = floor(h);
          int y1 = ceil(h);
          T dist_x = w - x0, dist_y = h - y0;
          T q00 = (1 - dist_x) * (1 - dist_y);
          T q01 = (1 - dist_x) * dist_y;
          T q10 = dist_x * (1 - dist_y);
          T q11 = dist_x * dist_y;
          int bottom_index_base = c * height * width;
          AtomicAdd(offset_bottom_data_diff + bottom_index_base + y0 * width + x0, q00 * diff_val);
          AtomicAdd(offset_bottom_data_diff + bottom_index_base + y1 * width + x0, q01 * diff_val);
          AtomicAdd(offset_bottom_data_diff + bottom_index_base + y0 * width + x1, q10 * diff_val);
          AtomicAdd(offset_bottom_data_diff + bottom_index_base + y1 * width + x1, q11 * diff_val);

          if (no_trans) {
            continue;
          }
          T U00 = offset_bottom_data[bottom_index_base + y0 * width + x0];
          T U01 = offset_bottom_data[bottom_index_base + y1 * width + x0];
          T U10 = offset_bottom_data[bottom_index_base + y0 * width + x1];
          T U11 = offset_bottom_data[bottom_index_base + y1 * width + x1];
          T diff_x = (U11 * dist_y + U10 * (1 - dist_y) - U01 * dist_y - U00 * (1 - dist_y)) * trans_std * diff_val;
          diff_x *= roi_width;
          T diff_y = (U11 * dist_x + U01 * (1 - dist_x) - U10 * dist_x - U00 * (1 - dist_x)) * trans_std * diff_val;
          diff_y *= roi_height;

          AtomicAdd(bottom_trans_diff + (((n * num_classes + class_id) * 2) * part_size + part_h) * part_size + part_w,
                    diff_x);
          AtomicAdd(
              bottom_trans_diff + (((n * num_classes + class_id) * 2 + 1) * part_size + part_h) * part_size + part_w,
              diff_y);
        }
      }
    }
  };
  d.parallelFor(count, Eigen::TensorOpCost(count, count, count), f);
}

template<typename T>
void DeformablePSROIPoolForwardCpuKernel(const CPUDevice &d, const int count, const T *bottom_data,
                                         const T spatial_scale, const int channels,
                                         const int height, const int width,
                                         const int pooled_height, const int pooled_width,
                                         const T *bottom_rois, const T *bottom_trans,
                                         const int no_trans, const T trans_std,
                                         const int sample_per_part, const int output_dim,
                                         const int group_size, const int part_size,
                                         const int num_classes, const int channels_each_class,
                                         T *top_data, T *top_count) {
  auto f = [count, bottom_data, spatial_scale, channels, height, width, pooled_height, pooled_width,
      bottom_rois, bottom_trans, no_trans, trans_std, sample_per_part, output_dim, group_size,
      part_size, num_classes, channels_each_class, top_data, top_count](int64 start, int64 end) {
    for (int64 index = start; index < end; ++index) {
      // The output is in order (n, ctop, ph, pw)
      int pw = index % pooled_width;
      int ph = (index / pooled_width) % pooled_height;
      int ctop = (index / pooled_width / pooled_height) % output_dim;
      int n = index / pooled_width / pooled_height / output_dim;
      // [start, end) interval for spatial sampling
      const T *offset_bottom_rois = bottom_rois + n * 5;
      int roi_batch_ind = offset_bottom_rois[0];
      T roi_start_w = (T) (round(offset_bottom_rois[1])) * spatial_scale - 0.5;
      T roi_start_h = (T) (round(offset_bottom_rois[2])) * spatial_scale - 0.5;
      T roi_end_w = (T) (round(offset_bottom_rois[3]) + 1.) * spatial_scale - 0.5;
      T roi_end_h = (T) (round(offset_bottom_rois[4]) + 1.) * spatial_scale - 0.5;
      // Force too small ROIs to be 1x1
      T roi_width = std::max(roi_end_w - roi_start_w, static_cast<T>(0.1)); // avoid 0
      T roi_height = std::max(roi_end_h - roi_start_h, static_cast<T>(0.1));
      // Compute w and h at bottom
      T bin_size_h = roi_height / static_cast<T>(pooled_height);
      T bin_size_w = roi_width / static_cast<T>(pooled_width);
      T sub_bin_size_h = bin_size_h / static_cast<T>(sample_per_part);
      T sub_bin_size_w = bin_size_w / static_cast<T>(sample_per_part);
      int part_h = floor(static_cast<T>(ph) / pooled_height * part_size);
      int part_w = floor(static_cast<T>(pw) / pooled_width * part_size);
      int class_id = ctop / channels_each_class;
      T trans_x = no_trans ? (T) (0) :
                  bottom_trans[(((n * num_classes + class_id) * 2) * part_size + part_h) * part_size + part_w]
                      * (T) trans_std;
      T trans_y = no_trans ? (T) (0) :
                  bottom_trans[(((n * num_classes + class_id) * 2 + 1) * part_size + part_h) * part_size + part_w]
                      * (T) trans_std;
      T wstart = static_cast<T>(pw) * bin_size_w + roi_start_w;
      wstart += trans_x * roi_width;
      T hstart = static_cast<T>(ph) * bin_size_h + roi_start_h;
      hstart += trans_y * roi_height;
      T sum = 0;
      int total = 0;
      int gw = floor(static_cast<T>(pw) * group_size / pooled_width);
      int gh = floor(static_cast<T>(ph) * group_size / pooled_height);
      gw = std::min(std::max(gw, 0), group_size - 1);
      gh = std::min(std::max(gh, 0), group_size - 1);
      const T *offset_bottom_data = bottom_data + (roi_batch_ind * channels) * height * width;
      for (int ih = 0; ih < sample_per_part; ++ih) {
        for (int iw = 0; iw < sample_per_part; ++iw) {
          T w = wstart + iw * sub_bin_size_w;
          T h = hstart + ih * sub_bin_size_h;
          // bilinear interpolation
          if (w < -0.5 || w > width - 0.5 || h < -0.5 || h > height - 0.5) {
            continue;
          }
          w = std::min(std::max(w, static_cast<T>(0.)), static_cast<T>(width - 1.));
          h = std::min(std::max(h, static_cast<T>(0.)), static_cast<T>(height - 1.));
          int c = (ctop * group_size + gh) * group_size + gw;
          T val = DmcnIm2colBilinear(offset_bottom_data + c * height * width, w, h, w, (T) height, (T) width);
          sum += val;
          total++;
        }
      }
      top_data[index] = total == 0 ? (T) (0) : sum / total;
      top_count[index] = total;
    }
  };
  d.parallelFor(count, Eigen::TensorOpCost(count, count, count), f);
}
template<typename DType>
void DeformableConv2DIm2ColCPUKernel(const CPUDevice &d,
                                     const int n,
                                     const DType *data_im,
                                     const DType *data_offset,
                                     const DType *data_mask,

                                     const int height,
                                     const int width,
                                     const int kernel_h,
                                     const int kernel_w,
                                     const int pad_h,
                                     const int pad_w,
                                     const int stride_h,
                                     const int stride_w,
                                     const int dilation_h,
                                     const int dilation_w,

                                     const int channel_per_deformable_group, // 输入图通道数除以deformable_group的数量,
                                     const int batch_size,
                                     const int num_channels,
                                     const int deformable_group, //这里的batch_size代表的是im2col_step_, 一般就设为1了
                                     const int height_col,
                                     const int width_col,
                                     DType *data_col) {
  auto f = [n, data_im, data_offset, data_mask, height, width, kernel_h, kernel_w,
      pad_h, pad_w, stride_w, stride_h, dilation_w, dilation_h, channel_per_deformable_group,
      batch_size, num_channels, deformable_group, height_col, width_col, data_col](int64 start, int64 end) {
    for (int64 index = start; index < end; index++) {
      const int w_col = index % width_col;
      const int h_col = (index / width_col) % height_col;
      const int b_col = (index / width_col / height_col) % batch_size;
      const int c_im = (index / width_col / height_col) / batch_size;
      const int c_col = c_im * kernel_h * kernel_w;

      // compute deformable group index
      const int deformable_group_index = c_im / channel_per_deformable_group;

      const int h_in = h_col * stride_h - pad_h;
      const int w_in = w_col * stride_w - pad_w;

      DType *data_col_ptr = data_col + ((c_col * batch_size + b_col) * height_col + h_col) * width_col + w_col;
      const DType *data_im_ptr = data_im + (b_col * num_channels + c_im) * height * width;
      const DType *data_offset_ptr = data_offset
          + (b_col * deformable_group + deformable_group_index) * 2 * kernel_h * kernel_w * height_col
              * width_col; //

      const DType *data_mask_ptr = data_mask
          + (b_col * deformable_group + deformable_group_index) * kernel_h * kernel_w * height_col * width_col; //
      for (int i = 0; i < kernel_h; ++i) {
        for (int j = 0; j < kernel_w; ++j) {
          const int data_offset_h_ptr = ((2 * (i * kernel_w + j)) * height_col + h_col) * width_col + w_col;
          const int data_offset_w_ptr = ((2 * (i * kernel_w + j) + 1) * height_col + h_col) * width_col + w_col;
          const int data_mask_hw_ptr = ((i * kernel_w + j) * height_col + h_col) * width_col + w_col;
          const DType offset_h = data_offset_ptr[data_offset_h_ptr];
          const DType offset_w = data_offset_ptr[data_offset_w_ptr];
          const DType mask = data_mask_ptr[data_mask_hw_ptr];
          auto val = static_cast<DType>(0);
          const DType h_im = h_in + i * dilation_h + offset_h;
          const DType w_im = w_in + j * dilation_w + offset_w;
          if (h_im > -1 && w_im > -1 && h_im < height && w_im < width) {
            val = DmcnIm2colBilinear(data_im_ptr, width, height, width, h_im, w_im);
          }
          *data_col_ptr = val * mask;
          data_col_ptr += batch_size * height_col * width_col;
        }
      }
    }
  };
  d.parallelFor(n, Eigen::TensorOpCost(n, n, n), f);
}

template<typename DType>
void DeformableConv2DCol2ImCPUKernel(const CPUDevice &d, const int n,
                                     const DType *data_col, const DType *data_offset, const DType *data_mask,
                                     const int channels, const int height, const int width,
                                     const int kernel_h, const int kernel_w,
                                     const int pad_h, const int pad_w,
                                     const int stride_h, const int stride_w,
                                     const int dilation_h, const int dilation_w,
                                     const int channel_per_deformable_group,
                                     const int batch_size, const int deformable_group,
                                     const int height_col, const int width_col,
                                     DType *grad_im) {
  auto f = [n, data_col, data_offset, data_mask, channels, height, width, kernel_h,
      kernel_w, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
      channel_per_deformable_group, batch_size, deformable_group, height_col, width_col, grad_im](int64 start,
                                                                                                  int64 end) {
    for (int64 index = start; index < end; ++index) {
      const int j = (index / width_col / height_col / batch_size) % kernel_w;
      const int i = (index / width_col / height_col / batch_size / kernel_w) % kernel_h;
      const int c = index / width_col / height_col / batch_size / kernel_w / kernel_h;
      // compute the start and end of the output
      const int deformable_group_index = c / channel_per_deformable_group;
      int w_out = index % width_col;
      int h_out = (index / width_col) % height_col;
      int b = (index / width_col / height_col) % batch_size;
      int w_in = w_out * stride_w - pad_w;
      int h_in = h_out * stride_h - pad_h;

      const DType *data_offset_ptr = data_offset
          + (b * deformable_group + deformable_group_index) * 2 * kernel_h * kernel_w * height_col * width_col;
      const DType *data_mask_ptr = data_mask
          + (b * deformable_group + deformable_group_index) * kernel_h * kernel_w * height_col * width_col;
      const int data_offset_h_ptr = ((2 * (i * kernel_w + j)) * height_col + h_out) * width_col + w_out;
      const int data_offset_w_ptr = ((2 * (i * kernel_w + j) + 1) * height_col + h_out) * width_col + w_out;
      const int data_mask_hw_ptr = ((i * kernel_w + j) * height_col + h_out) * width_col + w_out;
      const DType offset_h = data_offset_ptr[data_offset_h_ptr];
      const DType offset_w = data_offset_ptr[data_offset_w_ptr];
      const DType mask = data_mask_ptr[data_mask_hw_ptr];
      const DType cur_inv_h_data = h_in + i * dilation_h + offset_h;
      const DType cur_inv_w_data = w_in + j * dilation_w + offset_w;

      const DType cur_top_grad = data_col[index] * mask;
      const int cur_h = (int) cur_inv_h_data;
      const int cur_w = (int) cur_inv_w_data;
      for (int dy = -2; dy <= 2; dy++) {
        for (int dx = -2; dx <= 2; dx++) {
          if (cur_h + dy >= 0 && cur_h + dy < height &&
              cur_w + dx >= 0 && cur_w + dx < width &&
              abs(cur_inv_h_data - (cur_h + dy)) < 1 &&
              abs(cur_inv_w_data - (cur_w + dx)) < 1
              ) {
            int cur_bottom_grad_pos = ((b * channels + c) * height + cur_h + dy) * width + cur_w + dx;
            DType weight =
                DmcnGetGradientWeight(cur_inv_h_data, cur_inv_w_data, cur_h + dy, cur_w + dx, height, width);
            AtomicAdd(grad_im + cur_bottom_grad_pos, weight * cur_top_grad);
//                      *(grad_im + cur_bottom_grad_pos) += weight * cur_top_grad;
          }
        }
      }
    }
  };
  d.parallelFor(n, Eigen::TensorOpCost(n, n, n), f);
}
template<typename DType>
void DeformableConv2DCol2ImCoordCPUKernel(
    const CPUDevice &d,
    const int n,
    const DType *data_col, const DType *data_im,
    const DType *data_offset, const DType *data_mask,
    const int channels, const int height, const int width, // 输入的C, H, W
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int channel_per_deformable_group,
    const int batch_size, const int offset_channels, const int deformable_group,
    const int height_col, const int width_col,
    DType *grad_offset, DType *grad_mask) {
  auto f = [n, data_col, data_im, data_offset, data_mask, channels, height, width,
      kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
      channel_per_deformable_group, batch_size, offset_channels, deformable_group,
      height_col, width_col, grad_offset, grad_mask](int64 start, int64 end) {
    for (int64 index = start; index < end; index++) {
      DType val = 0, mval = 0;
      int w = index % width_col;
      int h = (index / width_col) % height_col;
      int c = (index / width_col / height_col) % offset_channels;
      int b = (index / width_col / height_col) / offset_channels;
      // compute the start and end of the output

      const int deformable_group_index = c / (2 * kernel_h * kernel_w);
      const int col_step = kernel_h * kernel_w;
      int cnt = 0;
      const DType *data_col_ptr =
          data_col + deformable_group_index * channel_per_deformable_group * batch_size * width_col * height_col;
      const DType *data_im_ptr = data_im
          + (b * deformable_group + deformable_group_index) * channel_per_deformable_group / kernel_h / kernel_w
              * height * width;
      const DType *data_offset_ptr = data_offset
          + (b * deformable_group + deformable_group_index) * 2 * kernel_h * kernel_w * height_col * width_col;
      const DType *data_mask_ptr = data_mask
          + (b * deformable_group + deformable_group_index) * kernel_h * kernel_w * height_col * width_col;

      const int offset_c = c - deformable_group_index * 2 * kernel_h * kernel_w;

      for (int col_c = (offset_c / 2); col_c < channel_per_deformable_group; col_c += col_step) {
        const int col_pos = (((col_c * batch_size + b) * height_col) + h) * width_col + w;
        const int bp_dir = offset_c % 2;

        int j = (col_pos / width_col / height_col / batch_size) % kernel_w;
        int i = (col_pos / width_col / height_col / batch_size / kernel_w) % kernel_h;
        int w_out = col_pos % width_col;
        int h_out = (col_pos / width_col) % height_col;
        int w_in = w_out * stride_w - pad_w;
        int h_in = h_out * stride_h - pad_h;
        const int data_offset_h_ptr = (((2 * (i * kernel_w + j)) * height_col + h_out) * width_col + w_out);
        const int data_offset_w_ptr = (((2 * (i * kernel_w + j) + 1) * height_col + h_out) * width_col + w_out);
        const int data_mask_hw_ptr = (((i * kernel_w + j) * height_col + h_out) * width_col + w_out);
        const DType offset_h = data_offset_ptr[data_offset_h_ptr];
        const DType offset_w = data_offset_ptr[data_offset_w_ptr];
        const DType mask = data_mask_ptr[data_mask_hw_ptr];
        DType inv_h = h_in + i * dilation_h + offset_h;
        DType inv_w = w_in + j * dilation_w + offset_w;
        if (inv_h <= -1 || inv_w <= -1 || inv_h >= height || inv_w >= width) {
          inv_h = inv_w = -2;
        } else {
          mval += data_col_ptr[col_pos]
              * DmcnIm2colBilinear(data_im_ptr + cnt * height * width, width, height, width, inv_h, inv_w);
        }
        const DType weight = DmcnGetCoordinateWeight(
            inv_h, inv_w,
            height, width, data_im_ptr + cnt * height * width, width, bp_dir);
        val += weight * data_col_ptr[col_pos] * mask;
        cnt += 1;
      }

      grad_offset[index] = val;
      // KERNEL_ASSIGN(grad_offset[index], offset_req, val);
      if (offset_c % 2 == 0) {
        grad_mask[
            (((b * deformable_group + deformable_group_index) * kernel_h * kernel_w + offset_c / 2) * height_col
                + h) * width_col + w] = mval;
        // KERNEL_ASSIGN(grad_mask[(((b * deformable_group + deformable_group_index) * kernel_h * kernel_w + offset_c / 2) * height_col + h) * width_col + w], mask_req, mval);
      }
    }
  };
  d.parallelFor(n, Eigen::TensorOpCost(n, n, n), f);
}
template<typename DType>
void PureAddToKernel(const CPUDevice &d, const int n, DType *result_data, const DType *right_data) {
  auto f = [n, result_data, right_data](int64 start, int64 end) {
    for (int64 index = start; index < end; index++) {
      *(result_data + index) += (right_data[index]);
    }
  };
  d.parallelFor(n, Eigen::TensorOpCost(n, n, n), f);
}
template<typename DType>
void SetZeroKernel(const CPUDevice &d, const int n, DType *result_data) {
  auto f = [n, result_data](int64 start, int64 end) {
    for (int64 index = start; index < end; ++index) {
      *(result_data + index) = DType(0);
    }
  };
  d.parallelFor(n, Eigen::TensorOpCost(n, n, n), f);
}
template<typename DType>
void SetOneKernel(const CPUDevice &d, const int n, DType *result_data) {
  auto f = [n, result_data](int64 start, int64 end) {
    for (int64 index = start; index < end; ++index) {
      *(result_data + index) = DType(1);
    }
  };
  d.parallelFor(n, Eigen::TensorOpCost(n, n, n), f);
}

template<typename DType>
void DeformableConv2DCol2ImCoord<CPUDevice, DType>::operator()(const Eigen::ThreadPoolDevice &d,
                                                               const DType *data_col,
                                                               const DType *data_im,
                                                               const DType *data_offset,
                                                               const DType *data_mask,
                                                               const TShape &im_shape,
                                                               const TShape &col_shape,
                                                               const TShape &kernel_shape,
                                                               const TShape &pad,
                                                               const TShape &stride,
                                                               const TShape &dilation,
                                                               const int32_t deformable_group,
                                                               DType *grad_offset,
                                                               DType *grad_mask) {
  int num_spatial_axes = kernel_shape.size();
  int num_kernels =
      col_shape[1] * col_shape[2] * col_shape[3] * 2 * kernel_shape[0] * kernel_shape[1] * deformable_group;
  int channel_per_deformable_group = col_shape[0] / deformable_group;
  switch (num_spatial_axes) {
    case 2:
      DeformableConv2DCol2ImCoordCPUKernel<DType>(d,
                                                  num_kernels,
                                                  data_col,
                                                  data_im,
                                                  data_offset,
                                                  data_mask,
                                                  im_shape[1],
                                                  im_shape[2],
                                                  im_shape[3],
                                                  kernel_shape[0],
                                                  kernel_shape[1],
                                                  pad[0],
                                                  pad[1],
                                                  stride[0],
                                                  stride[1],
                                                  dilation[0],
                                                  dilation[1],
                                                  channel_per_deformable_group,
                                                  col_shape[1],
                                                  2 * kernel_shape[0] * kernel_shape[1] * deformable_group,
                                                  deformable_group,
                                                  col_shape[2],
                                                  col_shape[3],
                                                  grad_offset,
                                                  grad_mask);
      break;
    default:LOG(FATAL) << "col2im_nd_gpu does not support computation with "
                       << num_spatial_axes << "spatial axes";
  }
}

template<typename DType>
void DeformableConv2DCol2Im<CPUDevice, DType>::operator()(const Eigen::ThreadPoolDevice &d,
                                                          const DType *data_col,
                                                          const DType *data_offset,
                                                          const DType *data_mask,
                                                          const TShape &im_shape,
                                                          const TShape &col_shape,
                                                          const TShape &kernel_shape,
                                                          const TShape &pad,
                                                          const TShape &stride,
                                                          const TShape &dilation,
                                                          const int32_t deformable_group,
                                                          DType *grad_im) {
  int num_spatial_axes = kernel_shape.size();
  int channel_per_deformable_group = im_shape[1] / deformable_group;
  int num_kernels = ProdShape(col_shape, 0, col_shape.size());
  // num_axes should be smaller than block size
  //   using namespace mxnet_op;
  switch (num_spatial_axes) {
    case 2:
      // To avoid involving atomic operations, we will launch one kernel per
      // bottom dimension, and then in the kernel add up the top dimensions.
      // NOLINT_NEXT_LINE(whitespace/operators)
      DeformableConv2DCol2ImCPUKernel<DType>(
          d, num_kernels, data_col, data_offset, data_mask, im_shape[1], im_shape[2], im_shape[3],
          kernel_shape[0], kernel_shape[1], pad[0], pad[1], stride[0], stride[1],
          dilation[0], dilation[1], channel_per_deformable_group,
          col_shape[1], deformable_group, col_shape[2], col_shape[3], grad_im);
      break;
    default:LOG(FATAL) << "col2im_nd_gpu does not support computation with "
                       << num_spatial_axes << " spatial axes";
  }
}

template<typename DType>
void DeformableConv2DIm2Col<CPUDevice, DType>::operator()(const Eigen::ThreadPoolDevice &d,
                                                          const DType *data_im,
                                                          const DType *data_offset,
                                                          const DType *data_mask,
                                                          const TShape &im_shape,
                                                          const TShape &col_shape,
                                                          const TShape &kernel_shape,
                                                          const TShape &pad,
                                                          const TShape &stride,
                                                          const TShape &dilation,
                                                          const int32_t deformable_group,
                                                          DType *data_col) {
  int num_spatial_axes = kernel_shape.size();
  int channel_per_deformable_group = im_shape[1] / deformable_group; // imshape[1] = 输入图的通道数
  int num_kernels = im_shape[1] * ProdShape(col_shape,
                                            1,
                                            col_shape.size()); // K * N / k.Size(), k = filter, col_shape = [K, im2col_step_, H, W]
  switch (num_spatial_axes) {
    case 2:
      DeformableConv2DIm2ColCPUKernel<DType>(
          d,
          num_kernels,
          data_im,
          data_offset,
          data_mask,
          im_shape[2], im_shape[3],
          kernel_shape[0], kernel_shape[1],
          pad[0], pad[1],
          stride[0], stride[1],
          dilation[0], dilation[1],
          channel_per_deformable_group,
          col_shape[1], im_shape[1],
          deformable_group,
          col_shape[2], col_shape[3],
          data_col);
      // MSHADOW_CUDA_POST_KERNEL_CHECK(modulated_deformable_im2col_gpu_kernel);
      break;
    default:LOG(FATAL) << "im2col_nd_gpu does not support computation with "
                       << num_spatial_axes << " spatial axes";
  }
}

template<typename DType>
void SetZeros<CPUDevice, DType>::operator()(const Eigen::ThreadPoolDevice &d, int n, DType *result_data) {
  SetZeroKernel(d, n, result_data);
}
template<typename DType>
void PureAddTo<CPUDevice, DType>::operator()(const Eigen::ThreadPoolDevice &d,
                                             const int n,
                                             DType *result_data,
                                             const DType *right_data) {
  PureAddToKernel(d, n, result_data, right_data);
}
template<typename DType>
void SetOne<CPUDevice, DType>::operator()(const Eigen::ThreadPoolDevice &d, int n, DType *result_data) {
  SetOneKernel(d, n, result_data);
}
template<typename DType>
void SetNumAtIndex<CPUDevice, DType>::operator()(const Eigen::ThreadPoolDevice &d, DType num, int index, DType *data) {
  *(data + index) = num;
}

template<typename T>
void LaunchBatchMatMul<CPUDevice, T>::launch(OpKernelContext *context,
                                             const TensorShape &in_x_shape,
                                             const TensorShape &in_y_shape,
                                             const T *in_x_ptr,
                                             const T *in_y_ptr,
                                             bool adj_x,
                                             bool adj_y,
                                             T *out) {
  const int64 m = in_x_shape.dim_size(adj_x ? 2 : 1);
  const int64 k = in_x_shape.dim_size(adj_x ? 1 : 2);
  const int64 n = in_y_shape.dim_size(adj_y ? 1 : 2);
  const uint64 batch_size = in_x_shape.dim_size(0);
  Eigen::TensorMap<Eigen::Tensor<const T, 3, Eigen::RowMajor>>
      t_in_x(in_x_ptr, in_x_shape.AsEigenDSizes<3, Eigen::DenseIndex>());
  Eigen::TensorMap<Eigen::Tensor<const T, 3, Eigen::RowMajor>>
      t_in_y(in_y_ptr, in_y_shape.AsEigenDSizes<3, Eigen::DenseIndex>());
  Eigen::TensorMap<Eigen::Tensor<T, 3, Eigen::RowMajor>> t_out(out, batch_size, m, n);
  Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> contract_pairs;
  contract_pairs[0] = ContractionDims(adj_x, adj_y);
  auto &device = context->eigen_device<CPUDevice>();
  for (int i = 0; i < t_out.dimension(0); ++i) {
    t_out.template chip<0>(i).device(device) =
        (t_in_x.template chip<0>(i)).template contract(t_in_y.template chip<0>(i), contract_pairs);
  }
}

template<typename T>
void DeformablePSROIPoolForward<CPUDevice, T>::operator()(const CPUDevice &d,
                                                          const int count,
                                                          const T *bottom_data,
                                                          const T spatial_scale,
                                                          const int channels,
                                                          const int height,
                                                          const int width,
                                                          const int pooled_height,
                                                          const int pooled_width,
                                                          const T *bottom_rois,
                                                          const T *bottom_trans,
                                                          const int no_trans,
                                                          const T trans_std,
                                                          const int sample_per_part,
                                                          const int output_dim,
                                                          const int group_size,
                                                          const int part_size,
                                                          const int num_classes,
                                                          const int channels_each_class,
                                                          T *top_data,
                                                          T *top_count) {
  DeformablePSROIPoolForwardCpuKernel<T>(d,
                                         count,
                                         bottom_data,
                                         spatial_scale,
                                         channels,
                                         height,
                                         width,
                                         pooled_height,
                                         pooled_width,
                                         bottom_rois,
                                         bottom_trans,
                                         no_trans,
                                         trans_std,
                                         sample_per_part,
                                         output_dim,
                                         group_size,
                                         part_size,
                                         num_classes,
                                         channels_each_class,
                                         top_data,
                                         top_count);
}

template<typename T>
void DeformablePSROIPoolBackwardKernel<CPUDevice, T>::operator()(const CPUDevice &d,
                                                                 const int count,
                                                                 const T *top_diff,
                                                                 const T *top_count,
                                                                 const int num_rois,
                                                                 const T spatial_scale,
                                                                 const int channels,
                                                                 const int height,
                                                                 const int width,
                                                                 const int pooled_height,
                                                                 const int pooled_width,
                                                                 const int output_dim,
                                                                 T *bottom_data_diff,
                                                                 T *bottom_trans_diff,
                                                                 const T *bottom_data,
                                                                 const T *bottom_rois,
                                                                 const T *bottom_trans,
                                                                 const int no_trans,
                                                                 const T trans_std,
                                                                 const int sample_per_part,
                                                                 const int group_size,
                                                                 const int part_size,
                                                                 const int num_classes,
                                                                 const int channels_each_class) {
  DeformablePSROIPoolBackwardCpuAccKernel<T>(d,
                                             count,
                                             top_diff,
                                             top_count,
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
template
struct DeformableConv2DIm2Col<CPUDevice, double>;
template
struct DeformableConv2DCol2Im<CPUDevice, double>;
template
struct DeformableConv2DCol2ImCoord<CPUDevice, double>;
template
struct PureAddTo<CPUDevice, double>;
template
struct SetOne<CPUDevice, double>;
template
struct SetZeros<CPUDevice, double>;
template
struct SwapAxis<CPUDevice, double>;
template
struct SetNumAtIndex<CPUDevice, double>;

template
struct DeformableConv2DIm2Col<CPUDevice, float>;
template
struct DeformableConv2DCol2Im<CPUDevice, float>;
template
struct DeformableConv2DCol2ImCoord<CPUDevice, float>;
template
struct PureAddTo<CPUDevice, float>;
template
struct SetOne<CPUDevice, float>;
template
struct SetZeros<CPUDevice, float>;
template
struct SwapAxis<CPUDevice, float>;
template
struct SetNumAtIndex<CPUDevice, float>;

template
struct LaunchBatchMatMul<CPUDevice, float>;
template
struct LaunchBatchMatMul<CPUDevice, double>;
template
struct DeformablePSROIPoolForward<CPUDevice, float>;
template
struct DeformablePSROIPoolForward<CPUDevice, double>;
template
struct DeformablePSROIPoolBackwardKernel<CPUDevice, float>;
template
struct DeformablePSROIPoolBackwardKernel<CPUDevice, double>;

}


