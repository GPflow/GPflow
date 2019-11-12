# Copyright 2019 GPflow Authors
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
"""Custom Extension for Sphinx to generate an documented API for sphinx"""
import gpflow
import generate_module_rst

def builder_inited(app):
    """This event runs as the builder is inited, and generates the RST"""
    generate_module_rst.set_global_path(app.srcdir)
    generate_module_rst.traverse_module_bfs([(gpflow, 0)], set([id(gpflow)]))

def setup(app):
    """This is used to set up the custom gpflow extension to generate our api"""
    app.setup_extension('sphinx.ext.autodoc')
    app.connect('builder-inited', builder_inited)
    return {'version': 1.0, 'parallel_read_safe': True}
