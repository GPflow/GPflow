// Copyright 2016 Mark van der Wilk
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


REGISTER_OP("TriToVec")
.Attr("T: realnumbertype")
.Input("trimat: T")
.Output("vec: T")
.Doc(R"doc(
Converts a series of triangular matrices to a series of vectors (i.e. a matrix). 

If the input is D x N x N, then the output is D x M, where the lower
triangle of each N x N matrix has been packed into an M-vector.
)doc");

using namespace tensorflow;

template <typename T>
class TriToVecOp : public OpKernel {
public:
  explicit TriToVecOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);

    const TensorShape& input_shape = input_tensor.shape();
    const int rank = input_shape.dims();

    // For now, keep it as just a matrix
    OP_REQUIRES(context, rank == 3,
		errors::InvalidArgument("TriToVec expects a rank-3 tensor, received shape: ",
		input_shape.DebugString()));

    const int k = input_shape.dim_size(rank - 1);  // Matrix size
    OP_REQUIRES(context, k == input_shape.dim_size(rank - 2),
        errors::InvalidArgument("input's last two dimensions must be equal, received shape: ",
        input_shape.DebugString()));

    auto f = input_tensor.flat_inner_dims<T, 3>();

    // Create an output tensor
    TensorShape out_shape({input_shape.dim_size(rank - 3), k * (k+1) / 2});
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape,
	                                                 &output_tensor));


    auto output = output_tensor->template flat<T>();
    int i = 0;
    for (int z = 0; z != f.dimension(0); z++) {
      for (int y = 0; y != f.dimension(1); y++) {
        for (int x = 0; x != f.dimension(2); x++) {
          if (y >= x) {
            output(i) = f(z, y, x);
            i++;
          }
        }
      }
    }
  }
};

#define REGISTER_KERNEL(type)             \
  REGISTER_KERNEL_BUILDER(                \
      Name("TriToVec")                    \
      .Device(DEVICE_CPU)                 \
      .TypeConstraint<type>("T"),         \
      TriToVecOp<type>                    \
  );


TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNEL);

#undef REGISTER_KERNEL
