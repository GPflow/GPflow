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


REGISTER_OP("RemoveRowElementsGrad")
.Attr("T: realnumbertype")
.Input("grad_mat: T")
.Input("grad_vec: T")
.Input("index: int32")
.Output("output_mat: T")
.Doc(R"doc(
A gradient method for RemoveRowElement. The same thing in reverse (injects zeros where needed)
)doc");

using namespace tensorflow;

template <typename T>
class RemoveRowElementsGradOp : public OpKernel {
public:
  explicit RemoveRowElementsGradOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensors
    const Tensor& gradmat_tensor = context->input(0);
    const Tensor& gradvec_tensor = context->input(1);
    const Tensor& index_tensor = context->input(2);

    const TensorShape& gradmat_shape = gradmat_tensor.shape();
    const TensorShape& gradvec_shape = gradmat_tensor.shape();
    const TensorShape& index_shape = index_tensor.shape();
    const int gradmat_rank = gradmat_shape.dims();
    const int gradvec_rank = gradmat_shape.dims();
    const int index_rank = index_shape.dims();
    const int num_rows = gradmat_shape.dim_size(0);
    const int num_index = index_shape.dim_size(0);

    // make sure ogradmat matrix is 2D
    OP_REQUIRES(context, gradmat_rank == 2,
		errors::InvalidArgument("RemoveRowElementsGrad expects a rank-2 tensor, received shape: ",
		gradmat_shape.DebugString()));
    // make sure index is 1D
    OP_REQUIRES(context, index_rank == 1,
		errors::InvalidArgument("RemoveRowElementsGrad expects a rank-1 index-tensor, received shape: ",
		index_shape.DebugString()));
    // make sure size of index and matrix match (TODO: better error string)
    OP_REQUIRES(context, num_index == num_rows,
		errors::InvalidArgument("RemoveRowElementsGrad requires the index to match the size of the matrix, received shapes: ",
		gradmat_shape.DebugString()));
    // make sure 


    // Create an output tensor
    TensorShape out_shape({gradmat_shape.dim_size(0), gradmat_shape.dim_size(1) + 1});
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape,
	                                                 &output_tensor));

    // get eigen objects for nice indexing
    auto gradmat = gradmat_tensor.shaped<T,2>({gradmat_shape.dim_size(0), gradmat_shape.dim_size(1)});
    auto gradvec = gradvec_tensor.flat<T>();
    auto index = index_tensor.flat<int32>();
    auto output = output_tensor->template shaped<T,2>({gradmat_shape.dim_size(0), gradmat_shape.dim_size(1) + 1});

    // loop through rows, injecting zeros for indexed elements
    int j;
    for (int i=0; i<gradmat_shape.dim_size(0); i++) {
      j=0;
      for (; j<index(i); j++) {
        output(i, j) = gradmat(i, j);
      }
      output(i, j) = gradvec(i);
      j++;
      for (; j<gradmat_shape.dim_size(1)+1; j++) {
        output(i, j) = gradmat(i, j-1);
      }
    }
  }
};

#define REGISTER_KERNEL(type)             \
  REGISTER_KERNEL_BUILDER(                \
      Name("RemoveRowElementsGrad")           \
      .Device(DEVICE_CPU)                 \
      .TypeConstraint<type>("T"),         \
      RemoveRowElementsGradOp<type>           \
  );


TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNEL);

#undef REGISTER_KERNEL
