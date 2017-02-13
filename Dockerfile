# Copyright 2016 The GPflow authors
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#USAGE:
#To run from prebuilt version use:
#docker run -it -p 8888:8888 gpflow/gpflow

#To replicate build use:
#docker build -t gpflow/gpflow .

#Uses official Tensorflow docker for cpu only.
FROM tensorflow/tensorflow
RUN apt-get -y update
RUN apt-get -y install git
RUN git clone https://github.com/GPflow/GPflow /usr/local/GPflow
RUN cd /usr/local/GPflow && \
    python setup.py develop
#Clear TensorFlow demos and replace them with GPflow examples.
RUN rm /notebooks/*.*
RUN rm /notebooks/*
COPY doc/source/notebooks/*.* /notebooks/
COPY doc/source/notebooks/data/* /notebooks/data/
COPY LICENSE /notebooks
COPY README.md /notebooks
