// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//https://github.com/onnx/onnx/blob/master/docs/Operators.md#Scatter
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/common.h"

namespace onnxruntime {

class Scatter final : public OpKernel {
 public:
  Scatter(const OpKernelInfo& info) : OpKernel(info) {
    ORT_ENFORCE(info.GetAttr<int64_t>("axis", &axis_).IsOK(),
                "Missing/Invalid 'axis' attribute value");
  }
  ~Scatter() = default;
  Status Compute(OpKernelContext* context) const override;

 private:
  int64_t axis_;
};

ONNX_CPU_OPERATOR_KERNEL(
    Scatter,
    1,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes())
        .TypeConstraint("Tind", std::vector<MLDataType>{DataTypeImpl::GetTensorType<int32_t>(), DataTypeImpl::GetTensorType<int64_t>()}),
    Scatter);

template <class Tin>
Status CopyScatterData(const Tensor* data_input, const Tensor* indecies_input, const Tensor* updates_input,
                       const int64_t axis, Tensor* data_output) {
  const TensorShape& input_data_shape = data_input->Shape();
  const Tin* indices_data = indecies_input->template Data<Tin>();
  const auto num_indices = indecies_input->Shape().Size();
  for (int64_t i = 0; i < num_indices; ++i) {
    Tin idx = indices_data[i];
    if (idx < 0 || idx >= input_data_shape[axis]) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "indices element out of data bounds, idx=", idx,
                             " data_dim=", input_data_shape[axis]);
    }
  }

  const auto num_input_dims = input_data_shape.NumDimensions();
  const auto input_elements = input_data_shape.Size();
  const auto element_bytes = data_input->DataType()->Size();

  const uint8_t* src_base = reinterpret_cast<const uint8_t*>(data_input->DataRaw());
  uint8_t* dst_base = reinterpret_cast<uint8_t*>(data_output->MutableDataRaw());
  const bool is_string_type = data_input->DataType() == DataTypeImpl::GetType<std::string>();
  // Copy input to output first and then poke updates over it
  // TODO: How to copy and scatter at the same time?
  if (is_string_type) {
    const std::string* str_begin = data_input->template Data<std::string>();
    const std::string* str_end = str_begin + input_elements;
    std::string* dst = data_output->template MutableData<std::string>();
    std::copy(str_begin, str_end, dst);
  } else {
    memcpy(dst_base, src_base, input_elements * element_bytes);
  }

  // Now poke updates
  // TODO: Investigate how to re-use the input Tensor for the output
  std::vector<int64_t> dim_weights(input_data_shape.GetDims().size(), 0);
  dim_weights[num_input_dims - 1] = 1;
  for (size_t i = num_input_dims - 2; i >= 0; --i) {
    dim_weights[i] = input_data_shape[i] * dim_weights[i + 1];
  }
  // Initialize with zeros
  std::vector<int64_t> counters(input_data_shape.GetDims().size(), 0);
  const uint8_t* update_data = reinterpret_cast<const uint8_t*>(updates_input->DataRaw());
  auto current_ind = indices_data;
  const auto end_indices = indices_data + indecies_input->Shape().Size();
  while (current_ind != end_indices) {
    const Tin idx = *current_ind;
    // Compute offset to destination
    size_t dst_offset = 0;
    for (size_t i = 0; i < num_input_dims - 1; ++i) {
      if (i == size_t(axis)) {
        dst_offset += idx * dim_weights[i];
      } else {
        dst_offset += counters[i] * dim_weights[i];
      }
    }
    dst_offset += counters[num_input_dims - 1];

    if (is_string_type) {
      reinterpret_cast<std::string*>(dst_base)[dst_offset] =
          reinterpret_cast<const std::string*>(update_data)[idx];
    } else {
      // Copy the element
      auto dst_offset_bytes = dst_offset * element_bytes;
      auto update_offset_bytes = idx * element_bytes;
      memcpy(dst_base + dst_offset_bytes, update_data + update_offset_bytes, element_bytes);
    }
    // Increment the counters
    for (size_t i = num_input_dims - 1; i >= 0; --i) {
      auto val = ++counters[i];
      if (val == input_data_shape[i]) {
        // Should not carry on the biggest dim
        assert(i != 0);
        counters[i] = 0;
      } else {
        break;
      }
    }
    ++current_ind;
  }
  return Status::OK();
}

Status Scatter::Compute(OpKernelContext* context) const {
  auto data_input = context->Input<Tensor>(0);
  const auto& input_data_shape = data_input->Shape();
  const auto axis = HandleNegativeAxis(axis_, input_data_shape.NumDimensions());

  auto indices_input = context->Input<Tensor>(1);
  auto updates_input = context->Input<Tensor>(2);

  if (data_input->DataType() != updates_input->DataType()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "data type is different from updates type");
  }

  auto& indicies_dims = indices_input->Shape().GetDims();
  auto& updates_dims = updates_input->Shape().GetDims();
  if (indicies_dims.size() != updates_dims.size()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Indicies and updates must have the same number of dimensions");
  }

  for (size_t i = 0; i < indicies_dims.size(); ++i) {
    if (indicies_dims[i] != updates_dims[i]) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Indicies vs updates dimensions differs at position=", i,
                             " ", indicies_dims[i], " vs ", updates_dims[i]);
    }
  }

  auto data_output = context->Output(0, input_data_shape);

  MLDataType Tind_type = indices_input->DataType();
  if (Tind_type == DataTypeImpl::GetType<int32_t>()) {
    ORT_RETURN_IF_ERROR(CopyScatterData<int32_t>(data_input, indices_input, updates_input, axis, data_output));
  } else if (Tind_type == DataTypeImpl::GetType<int64_t>()) {
    ORT_RETURN_IF_ERROR(CopyScatterData<int64_t>(data_input, indices_input, updates_input, axis, data_output));
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Expecting indices to be either int32_t or int64_t");
  }

  return Status::OK();
}

}  // namespace onnxruntime
