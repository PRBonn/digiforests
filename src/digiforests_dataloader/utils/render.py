# MIT License
#
# Copyright (c) 2025 Meher Malladi, Luca Lobefaro, Tiziano Guadagnino, Cyrill Stachniss
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import numpy as np
import open3d as o3d

from .cloud import Cloud


def render(geoms: list):
    """
    Renders a list of objects. Iterates through and handles Cloud and Cylinder objects.
    Rest are left alone assuming they are o3d geoms.
    """
    vis = _get_window()
    render_geoms = []
    for geom in geoms:
        if isinstance(geom, Cloud):
            render_geoms.append(geom.tpcd.to_legacy())
        else:
            render_geoms.append(geom)
    assert render_geoms, f"empty {render_geoms}"
    for geom in render_geoms:
        vis.add_geometry(geom)
    vis.run()
    vis.destroy_window()


def _get_window():
    o3d_vis = o3d.visualization.Visualizer()
    o3d_vis.create_window(
        window_name="Open3D",
        width=1920,
        height=1080,
        left=50,
        top=50,
        visible=True,
    )
    # Change the render options
    render_options = o3d_vis.get_render_option()
    render_options.point_size = 1
    render_options.mesh_show_wireframe = False
    render_options.mesh_show_back_face = False
    render_options.background_color = np.asarray([1.0, 1.0, 1.0])
    render_options.show_coordinate_frame = False

    return o3d_vis
