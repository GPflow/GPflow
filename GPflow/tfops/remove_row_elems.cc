// Copyright 2016 James Hensman
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
// http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cmath>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"


REGISTER_OP("RemoveRowElements")
.Attr("T: realnumbertype")
.Input("input_mat: T")
.Input("index: int32")
.Output("output_mat: T")
.Doc(R"doc(
For each row in input_mat, removes one element as specified in index
)doc");

using namespace tensorflow;

template <typename T>
class RemoveRowElementsOp : public OpKernel {
public:
  explicit RemoveRowElementsOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensors
    const Tensor& input_tensor = context->input(0);
    const Tensor& index_tensor = context->input(1);

    const TensorShape& input_shape = input_tensor.shape();
    const TensorShape& index_shape = index_tensor.shape();
    const int input_rank = input_shape.dims();
    const int index_rank = index_shape.dims();
    const int num_rows = input_shape.dim_size(0);
    const int num_index = index_shape.dim_size(0);

    // make sure oinput matrix is 2D
    OP_REQUIRES(context, input_rank == 2,
		errors::InvalidArgument("RemoveRowElements expects a rank-2 tensor, received shape: ",
		input_shape.DebugString()));
    // make sure index is 1D
    OP_REQUIRES(context, index_rank == 1,
		errors::InvalidArgument("RemoveRowElements expects a rank-1 index-tensor, received shape: ",
		index_shape.DebugString()));
    // make sure size of index and matrix match (TODO: better error string)
    OP_REQUIRES(context, num_index == num_rows,
		errors::InvalidArgument("RemoveRowElements requires the index to match the size of the matrix, received shapes: ",
		input_shape.DebugString()));
    // make sure 


    // Create an output tensor
    TensorShape out_shape({input_shape.dim_size(0), input_shape.dim_size(1) - 1});
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape,
	                                                 &output_tensor));

    // get eigen objects for nice indexing
    auto input = input_tensor.shaped<T,2>({input_shape.dim_size(0), input_shape.dim_size(1)});
    auto index = index_tensor.flat<int32>();
    auto output = output_tensor->template shaped<T,2>({input_shape.dim_size(0), input_shape.dim_size(1) - 1});

    // loop through rows, dropping indexed elements
    int j;
    for (int i=0; i<input_shape.dim_size(0); i++) {
      j=0;
      for (; j<index(i); j++) {
        output(i, j) = input(i, j);
      }
      j++;
      for (; j<input_shape.dim_size(1); j++) {
        output(i, j-1) = input(i, j);
      }
    }
  }
};

#define REGISTER_KERNEL(type)             \
  REGISTER_KERNEL_BUILDER(                \
      Name("RemoveRowElements")           \
      .Device(DEVICE_CPU)                 \
      .TypeConstraint<type>("T"),         \
      RemoveRowElementsOp<type>           \
  );


TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNEL);

#undef REGISTER_KERNEL
