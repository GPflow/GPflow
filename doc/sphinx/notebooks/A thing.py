# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import tensorflow as tf
import gpflow.utilities.dimobjs as do

# %% [markdown]
# # Example
#
# First I create a tensor to demo with:

# %%
t = tf.constant([[1, 2, 3], [4, 5, 6]])
t

# %% [markdown]
# Then I create some dimension objects. Here one for height (`h`) and one for width (`w`). The string argument is only used for pretty printing / debugging - we can get rid of it if we want to.

# %%
w, h = do.dims("w,h")

# %% [markdown]
# We can use the dimensions to transpose our test tensor:

# %%
h + w >> t >> w + h

# %% [markdown]
# We can multiply dimensions with `*`, and `do.one` is a constant size 1 dimension:

# %%
h + w >> t >> h * w + do.one

# %% [markdown]
# We can inline an operator to do things other than reshape / transpose. Here I use `reduce_sum` over the `w` dimension:

# %%
t2 = h + w >> t >> do.reduce_sum >> h
t2

# %% [markdown]
# We can use `tile` to get the `w` dimension back:

# %%
h >> t2 >> do.tile >> w + h

# %% [markdown]
# You can use tuples to provide multiple intputs and outputs to the function / operator. Here's an example with einsum:

# %%
(batch,) = do.dimses("batch")
h, w = do.dims("h,w")
t1 = tf.random.normal(shape=(1, 2, 3, 4))
t2 = tf.random.normal(shape=(4, 3))
(batch + h + w >> t1, w + h >> t2) >> do.einsum >> batch

# %% [markdown]
# ## Related work
#
# * <https://github.com/arogozhnikov/einops>
# * <https://github.com/facebookresearch/torchdim>
