#ifndef CORE_FRAMEWORK_TENSOR_H
#define CORE_FRAMEWORK_TENSOR_H

#include <string>
#include <vector>

#include "core/framework/allocator.h"
#include "core/framework/data_types.h"
#include "core/protobuf/onnx-ml.pb.h"

using namespace onnx;

namespace Lotus {
class TensorShape {
 public:
  TensorShape();

  TensorShape(const std::vector<int64_t>& dims);

  TensorShape(const TensorShape& p_other);

  // Return the dimension specified by <p_idx>.
  const int64_t operator[](int p_idx) const;

  bool operator==(const TensorShape& p_other) const {
    return m_dims == p_other.m_dims;
  }

  bool operator!=(const TensorShape& p_other) const {
    return !(*this == p_other);
  }

  const int64_t NumDimensions() const {
    return m_dims.size();
  }

  // Return the total number of elements.
  int64_t Size() const;

  // Return a new TensorShape of the dimensions from dimstart to dimend.
  TensorShape Slice(int p_dimstart, int p_dimend) const;

 private:
  // We use negative numbers for unknown symbolic dimension. Each negative
  // number represents a unique symbolic dimension.
  // InlinedVector<int64_t, 4> dims_;
  std::vector<int64_t> m_dims;
};

struct BufferDeleter {
  // TODO: we may need consider the lifetime of alloc carefully
  // The alloc_ here is the allocator that used to allocate the buffer
  // And need go with the unique_ptr together. If it is using our internal
  // allocator, it is ok as our allocators are global managed. But if it
  // is provide by user, user need to be very careful about it.
  // A weak_ptr may be a choice to reduce the impact, but that require to
  // change our current allocator mgr to use shared_ptr. Will revisit it
  // later.
  IAllocator* alloc_;
  BufferDeleter() : alloc_(nullptr) {}
  BufferDeleter(IAllocator* alloc)
      : alloc_(alloc) {}

  void operator()(void* p) const {
    if (alloc_)
      alloc_->Free(p);
  }
};

typedef std::unique_ptr<void, BufferDeleter> BufferUniquePtr;
typedef void* BufferNakedPtr;

/*
    We want to keep tensor as simple as possible, it is just a placeholder for a piece of memory, with additional shape information.
    Memory is managered by Executor / Workspace, so Tensor just use it, won't do any allocation / release 
    */
class Tensor {
  friend class TensorUtil;
  friend class MLValue;
  friend class ExecutionFrame;

 public:
  enum BufferStrategy {
    UNKNOWN,
    PREALLOCATEDBUFFER,
    OWNEDBUFFER
  };

  // Create an empty tensor with float type.
  // empty tensor is a tensor with 1-d shape (0,), and 0 elements.
  Tensor();
  // Create a empty tensor with given type
  Tensor(MLDataType p_type);
  // Create tensor with given type, shape, pre-allocate memory and allocator info.
  Tensor(MLDataType p_type, const TensorShape& p_shape, BufferNakedPtr p_data, const AllocatorInfo& alloc, const int64_t offset = 0);
  Tensor(MLDataType p_type, const TensorShape& p_shape, BufferUniquePtr p_data, const AllocatorInfo& alloc, const int64_t offset = 0);

  // Copy constructor and assign op will just pass the shape and memory reference to another tensor.
  // No deep clone / copy happened.
  Tensor(const Tensor& src);
  Tensor& ShallowCopy(const Tensor& other);

  Tensor(Tensor&& other);

  Tensor& operator=(Tensor&& other);

  // Returns the data type.
  MLDataType dtype() const { return dtype_; }

  // Returns the shape of the tensor.
  const TensorShape& shape() const { return shape_; }

  // Returns the location of the tensor's memory
  const AllocatorInfo& location() const { return alloc_info_; }

  template <typename T>
  T* mutable_data() {
    //Type check
    LOTUS_ENFORCE(DataTypeImpl::GetType<T>() == dtype_, "Tensor type mismatch.");
    return static_cast<T*>(
        static_cast<void*>(static_cast<char*>(GetRaw()) + byte_offset_));
  }

  template <typename T>
  const T* data() const {
    //Type check
    LOTUS_ENFORCE(DataTypeImpl::GetType<T>() == dtype_, "Tensor type mismatch.");
    return static_cast<const T*>(
        static_cast<void*>(static_cast<char*>(GetRaw()) + byte_offset_));
  }
  // More API methods.
 private:
  void Init(MLDataType p_type,
            const TensorShape& p_shape,
            BufferStrategy strategy,
            BufferNakedPtr p_raw_data,
            const AllocatorInfo& alloc,
            const int64_t offset = 0);

  void* GetRaw() const {
    switch (buffer_strategy_) {
      case PREALLOCATEDBUFFER:
        return p_naked_data_;
      case OWNEDBUFFER:
        return p_unique_data_.get();
      default:
        if (shape_.Size() == 0)  //empty tensor
          return nullptr;
        else
          LOTUS_THROW("Unknown buffer strategy!");
    }
  }

  BufferNakedPtr p_naked_data_;
  BufferUniquePtr p_unique_data_;
  BufferStrategy buffer_strategy_;

  TensorShape shape_;
  MLDataType dtype_;
  AllocatorInfo alloc_info_;
  int64_t byte_offset_;
};

}  // namespace Lotus

#endif  // CORE_FRAMEWORK_TENSOR_H