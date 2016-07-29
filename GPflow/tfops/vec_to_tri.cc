#include "tensorflow/core/framework/op.h"
#include <iostream>
#include <cmath>

REGISTER_OP("VecToTri")
.Input("vec: int32")
.Output("matrix: int32");

#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

class VecToTriOp : public OpKernel {
public:
  explicit VecToTriOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<int32>();

    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(input_tensor.shape()),
		errors::InvalidArgument("VecToTri expectsa 2-D matrix."));

    //std::cout << input_tensor.shape().dims() << std::endl;
    auto ds = input_tensor.shape().dim_sizes();
    //std::cout << ds[0] << " " << ds[1] << std::endl;

    int matsize = (int)std::floor(std::sqrt(ds[1] * 8 + 1) / 2.0 - 0.5);
    int recvecsize = (int)std::round(0.5*matsize*(matsize+1));
    //std::cout << matsize << std::endl;
    //std::cout << recvecsize << std::endl;

    OP_REQUIRES(context, recvecsize == ds[1],
		errors::InvalidArgument("Must have triangle number")
		);
    
    // Create an output tensor
    TensorShape out_shape({ds[0], matsize, matsize});
    for (int d = 0; d != out_shape.dims(); d++) {
      std::cout << out_shape.dim_size(d) << " ";
    }
    std::cout << std::endl;
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape,
						     &output_tensor));

    auto output = output_tensor->template flat<int32>();
    const int N = output.size();
    for (int i = 0; i != N; i++) {
      int output_mat = i / (matsize * matsize);
      int x = i % matsize;
      int y = (i / matsize) % matsize;
      // output(i) = (output_mat + 1) * 100 + y * 10 + x;  // For testing purposes
      if (x > y) {
	output(i) = 0;
      } else {
	int idx = (i % (matsize*matsize)) - (int)std::round(matsize*y-0.5*y*y-0.5*y);
	// output(i) = idx + ds[1] * output_mat;
	output(i) = input(idx + ds[1] * output_mat);
      }
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("VecToTri").Device(DEVICE_CPU), VecToTriOp);

