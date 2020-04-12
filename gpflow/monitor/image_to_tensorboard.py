# Copyright 2020 GPflow authors
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
""" ImageToTensorBoard class (depends on matplotlib being installed) """

from abc import ABC, abstractmethod
from io import BytesIO
from typing import Any, Callable, Dict, Optional

import numpy as np
import tensorflow as tf

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Axes, Figure

from .tensorboard import ToTensorBoard


__all__ = ["ImageToTensorBoard"]


class ImageToTensorBoard(ToTensorBoard):
    def __init__(
        self,
        log_dir: str,
        plotting_function: Callable[[Figure, Axes], Figure],
        name: Optional[str] = None,
        *,
        fig_kw: Optional[Dict[str, Any]] = None,
        subplots_kw: Optional[Dict[str, Any]] = None,
    ):
        """
        :param log_dir: directory in which to store the tensorboard files.
            Can be a nested: for example, './logs/my_run/'.
        :param plotting_function: function performing the plotting.
        :param name: name used in TensorBoard.
        :params fig_kw: Keywords to be passed to Figure constructor, such as `figsize`.
        :params subplots_kw: Keywords to be passed to figure.subplots constructor, such as
            `nrows`, `ncols`, `sharex`, `sharey`. By default the default values 
            from matplotlib.pyplot as used.
        """
        super().__init__(log_dir)
        self.plotting_function = plotting_function
        self.name = name
        self.fig_kw = fig_kw or {}
        self.subplots_kw = subplots_kw or {}

        self.fig = Figure(**self.fig_kw)
        if self.subplots_kw != {}:
            self.axes = self.fig.subplots(**self.subplots_kw)
        else:
            self.axes = self.fig.add_subplot(111)

    def _clear_axes(self):
        if isinstance(self.axes, np.ndarray):
            for ax in self.axes.flatten():
                ax.clear()
        else:
            self.axes.clear()

    def run(self, **unused_kwargs):
        self._clear_axes()
        self.plotting_function(self.fig, self.axes)
        canvas = FigureCanvasAgg(self.fig)
        canvas.draw()

        # get PNG data from the figure
        png_buffer = BytesIO()
        canvas.print_png(png_buffer)
        png_encoded = png_buffer.getvalue()
        png_buffer.close()

        image_tensor = tf.io.decode_png(png_encoded)[None]

        # Write to TensorBoard
        tf.summary.image(self.name, image_tensor, step=self.current_step)
