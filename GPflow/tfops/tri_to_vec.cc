#include <cmath>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"


REGISTER_OP("TriToVec")
.Attr("T: realnumbertype")
.Input("trimat: T")
.Output("vec: T");

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
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(input_shape),
		errors::InvalidArgument("TriToVec expects at least a 2-dim matrix, received shape: ",
		input_shape.DebugString()));

    const int k = input_shape.dim_size(rank - 1);
    OP_REQUIRES(context, k == input_shape.dim_size(rank - 2),
        errors::InvalidArgument("input's last two dimensions must be equal, received shape: ",
        input_shape.DebugString()));

    auto f = input_tensor.flat_inner_dims<T, 2>();

    // Create an output tensor
    TensorShape out_shape({k * (k+1) / 2});
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape,
	                                                 &output_tensor));

    auto output = output_tensor->template flat<T>();
    int i = 0;
    for (int y = 0; y != f.dimension(1); y++) {
      for (int x = 0; x != f.dimension(0); x++) {
        if (y >= x) {
          output(i) = f(y, x);
          i++;
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
