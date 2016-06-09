# Release 0.1.3
 - Removed the need for a fork of TensorFlow. Some of our bespoke ops are replaced by equivalent versions. 

# Release 0.1.2
 - Included the ability to compute the full covaraince matrix at predict time. See `GPModel.predict_f`
 - Included the ability to sample from the posterior function values. See `GPModel.predict_f_samples`
 - Unified code in conditionals.py: see deprecations in `gp_predict`, etc.
 - Added SGPR method (Sparse GP Regression)

# Release 0.1.1
 -  included the ability to use tensorflow's optimizers as well as the scipy ones
# Release 0.1.0
The initial release of GPflow. 
