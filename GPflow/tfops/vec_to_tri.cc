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


REGISTER_OP("VecToTri")
.Attr("T: realnumbertype")
.Input("vec: T")
.Output("matrix: T")
.Doc(R"doc(
Converts a matrix into a series of triangular matrices.

If the input is D x M, then the output is D x N x N, where the lower
triangle of each N x N matrix is constructed by unpacking each M-vector.

See also: TriToVec.
)doc");



using namespace tensorflow;

template <typename T>
class VecToTriOp : public OpKernel {
public:
  explicit VecToTriOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<T>();

    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(input_tensor.shape()),
		errors::InvalidArgument("VecToTri expects a 2-D matrix."));

    auto ds = input_tensor.shape().dim_sizes();

    int matsize = (int)std::floor(std::sqrt(ds[1] * 8 + 1) / 2.0 - 0.5);  // Deduce square matrix size
    int recvecsize = (int)std::round(0.5*matsize*(matsize+1));            // Reconstruct number of required vector elements

    OP_REQUIRES(context, recvecsize == ds[1],
	        errors::InvalidArgument("Must have triangle number of elements in the input vector.")
	        );
    
    // Create an output tensor
    TensorShape out_shape({ds[0], matsize, matsize});
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape,
						     &output_tensor));

    auto output = output_tensor->template flat<T>();
    const int N = output.size();
    for (int i = 0; i != N; i++) {
      int output_mat = i / (matsize * matsize);
      int x = i % matsize;
      int y = (i / matsize) % matsize;
      if (x > y) {
        output(i) = (T)0;
      } else {
        int idx = (i % (matsize*matsize)) - (int)std::round(matsize*y-0.5*y*y-0.5*y);
        output(i) = input(idx + ds[1] * output_mat);
      }
    }
  }
};

#define REGISTER_KERNEL(type)             \
  REGISTER_KERNEL_BUILDER(                \
      Name("VecToTri")                    \
      .Device(DEVICE_CPU)                 \
      .TypeConstraint<type>("T"),         \
      VecToTriOp<type>                    \
  );


TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNEL);

#undef REGISTER_KERNEL
