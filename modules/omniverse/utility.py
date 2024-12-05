# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import os
import vtk
from pxr import Usd, UsdGeom, Gf, Sdf
import trimesh
import numpy as np
import matplotlib.pyplot as plt


def convert_to_mesh(
    segmentation_path, output_folder, filename, label_value=1, smoothing_factor=0.5, reduction_ratio=0.0
):
    """
    Function to perform segmentation-to-mesh conversion and smoothing
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Generate 16 distinct colors using a colormap
    colormap = plt.get_cmap("tab20")  # Using tab20 for distinct colors
    colors = [colormap(i) for i in np.linspace(0, 1, 16)]

    # Step 1: Load segmentation (binary labelmap, e.g., NRRD file)
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(segmentation_path)
    reader.Update()

    label_values = [label_value] if isinstance(label_value, int) else label_value
    if len(label_values) > 1:
        renderer = vtk.vtkRenderer()
        render_window = vtk.vtkRenderWindow()
        render_window.AddRenderer(renderer)
    for i in label_values:
        # Step 2: Create Closed Surface Representation using vtkDiscreteFlyingEdges3D
        flying_edges = vtk.vtkDiscreteFlyingEdges3D()
        flying_edges.SetInputConnection(reader.GetOutputPort())
        flying_edges.ComputeGradientsOff()
        flying_edges.ComputeNormalsOff()
        flying_edges.SetValue(0, i)
        flying_edges.Update()

        if flying_edges.GetOutput().GetNumberOfPoints() == 0:
            print(f"No points found for label {i}. Skipping...")
            continue

        # Step 3: Decimate the mesh
        if reduction_ratio > 0.0:
            decimation_filter = vtk.vtkDecimatePro()
            decimation_filter.SetInputConnection(flying_edges.GetOutputPort())
            decimation_filter.SetFeatureAngle(60)
            decimation_filter.SplittingOff()
            decimation_filter.PreserveTopologyOn()
            decimation_filter.SetMaximumError(1)
            decimation_filter.SetTargetReduction(reduction_ratio)
            decimation_filter.Update()

        # Step 4: Smooth the resulting mesh
        smoothing_filter = vtk.vtkWindowedSincPolyDataFilter()
        numberOfIterations = int(20 + smoothing_factor * 40)
        passBand = pow(10.0, -4.0 * smoothing_factor)
        if reduction_ratio > 0.0:
            smoothing_filter.SetInputConnection(decimation_filter.GetOutputPort())
        else:
            smoothing_filter.SetInputConnection(flying_edges.GetOutputPort())
        smoothing_filter.SetNumberOfIterations(numberOfIterations)
        smoothing_filter.SetPassBand(passBand)
        smoothing_filter.BoundarySmoothingOff()
        smoothing_filter.FeatureEdgeSmoothingOff()
        smoothing_filter.NonManifoldSmoothingOn()
        smoothing_filter.NormalizeCoordinatesOn()
        smoothing_filter.Update()

        # Step 5: Generate normals for better shading
        # normals_filter = vtk.vtkPolyDataNormals()
        # normals_filter.SetInputConnection(smoothing_filter.GetOutputPort())
        # normals_filter.SplittingOff()
        # normals_filter.ConsistencyOn()
        # normals_filter.Update()

        # Step 6: Decimate the mesh further
        decimation = vtk.vtkQuadricDecimation()
        decimation.SetInputConnection(smoothing_filter.GetOutputPort())
        decimation.SetTargetReduction(0.9)  # 90% reduction, the same as slicer
        decimation.VolumePreservationOn()
        decimation.Update()

        # Step 7: Generate normals for better shading
        decimatedNormals = vtk.vtkPolyDataNormals()
        decimatedNormals.SetInputConnection(decimation.GetOutputPort())
        decimatedNormals.SplittingOff()
        decimatedNormals.ConsistencyOn()
        decimatedNormals.Update()

        # Step 8: convert to LPS
        ras2lps = vtk.vtkMatrix4x4()
        ras2lps.SetElement(0, 0, -1)
        ras2lps.SetElement(1, 1, -1)
        ras2lpsTransform = vtk.vtkTransform()
        ras2lpsTransform.SetMatrix(ras2lps)
        transformer = vtk.vtkTransformPolyDataFilter()
        transformer.SetTransform(ras2lpsTransform)
        transformer.SetInputConnection(decimatedNormals.GetOutputPort())
        transformer.Update()

        if len(label_values) > 1:
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(transformer.GetOutput())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            colorRGB = colors[i - 1][:3] if i < 15 else colors[i - 2][:3]
            colorHSV = [0, 0, 0]
            vtk.vtkMath.RGBToHSV(colorRGB, colorHSV)
            colorHSV[1] = min(colorHSV[1] * 1.5, 1.0)  # increase saturation
            colorHSV[2] = min(colorHSV[2] * 1.0, 1.0)  # increase brightness
            colorRGB = [0, 0, 0]
            vtk.vtkMath.HSVToRGB(colorHSV, colorRGB)
            actor.GetProperty().SetColor(colorRGB[0], colorRGB[1], colorRGB[2])
            actor.GetProperty().SetInterpolationToGouraud()
            renderer.AddActor(actor)

    output_filename = os.path.join(output_folder, filename)
    if len(label_values) > 1:
        exporter = vtk.vtkGLTFExporter()
        exporter.SetFileName(output_filename)
        exporter.SetRenderWindow(render_window)
        exporter.SetInlineData(True)
        exporter.Write()
    else:
        if flying_edges.GetOutput().GetNumberOfPoints() > 0:
            polydata = transformer.GetOutput()
            writer = vtk.vtkOBJWriter()
            writer.SetFileName(output_filename)
            writer.SetInputData(polydata)
            writer.Write()

    print(f"Mesh successfully exported to {output_filename}")


def convert_obj_to_usd(obj_filename, usd_filename):
    """
    Function to convert an OBJ file to USD
    """
    # Create a new USD stage
    stage = Usd.Stage.CreateNew(usd_filename)

    # Define a mesh at the root of the stage
    mesh = UsdGeom.Mesh.Define(stage, "/RootMesh")

    # Lists to hold OBJ data
    vertices = []
    normals = []
    texcoords = []
    face_vertex_indices = []
    face_vertex_counts = []

    # Mapping for OBJ indices (since they can be specified per face-vertex)
    normal_indices = []
    texcoord_indices = []

    # Read the OBJ file
    with open(obj_filename, "r") as obj_file:
        for line in obj_file:
            if line.startswith("v "):
                # Vertex position
                _, x, y, z = line.strip().split()
                vertices.append((float(x), float(y), float(z)))
            elif line.startswith("vn "):
                # Vertex normal
                _, nx, ny, nz = line.strip().split()
                normals.append((float(nx), float(ny), float(nz)))
            elif line.startswith("vt "):
                # Texture coordinate
                _, u, v = line.strip().split()
                texcoords.append((float(u), float(v)))
            elif line.startswith("f "):
                # Face
                face_elements = line.strip().split()[1:]
                vertex_count = len(face_elements)
                face_vertex_counts.append(vertex_count)
                for elem in face_elements:
                    indices = elem.split("/")
                    # OBJ indices are 1-based; subtract 1 for 0-based indexing
                    vi = int(indices[0]) - 1
                    ti = int(indices[1]) - 1 if len(indices) > 1 and indices[1] else None
                    ni = int(indices[2]) - 1 if len(indices) > 2 and indices[2] else None
                    face_vertex_indices.append(vi)
                    if ni is not None:
                        normal_indices.append(ni)
                    if ti is not None:
                        texcoord_indices.append(ti)

    # Set the mesh's points
    mesh.CreatePointsAttr([Gf.Vec3f(*v) for v in vertices])

    # Set the face vertex indices and counts
    mesh.CreateFaceVertexIndicesAttr(face_vertex_indices)
    mesh.CreateFaceVertexCountsAttr(face_vertex_counts)

    # Optionally set normals if they exist
    if normals and normal_indices:
        # Reorder normals according to face vertices
        ordered_normals = [normals[i] for i in normal_indices]
        mesh.CreateNormalsAttr([Gf.Vec3f(*n) for n in ordered_normals])
        mesh.SetNormalsInterpolation("faceVarying")  # Adjust based on how normals are specified

    # Optionally set texture coordinates if they exist
    if texcoords and texcoord_indices:
        # Reorder texcoords according to face vertices
        ordered_texcoords = [texcoords[i] for i in texcoord_indices]
        stPrimvar = mesh.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
        stPrimvar.Set([Gf.Vec2f(*tc) for tc in ordered_texcoords])

    # Save the stage
    stage.GetRootLayer().Save()

    print(f"USD file successfully exported to {usd_filename}")


def convert_mesh_to_usd(input_file, output_file):
    # Load the mesh
    mesh = trimesh.load(input_file)

    # Create a new USD stage
    stage = Usd.Stage.CreateNew(output_file)

    # If the mesh is a Scene, process each geometry
    if isinstance(mesh, trimesh.Scene):
        for name, geometry in mesh.geometry.items():
            # Create a unique path for each mesh
            mesh_path = f"/{name}"
            usd_mesh = UsdGeom.Mesh.Define(stage, mesh_path)

            # Set vertex positions
            usd_mesh.GetPointsAttr().Set([Gf.Vec3f(*vertex) for vertex in geometry.vertices])

            # Set face indices and counts
            face_vertex_indices = geometry.faces.flatten().tolist()
            face_vertex_counts = [len(face) for face in geometry.faces]

            usd_mesh.GetFaceVertexIndicesAttr().Set(face_vertex_indices)
            usd_mesh.GetFaceVertexCountsAttr().Set(face_vertex_counts)

            # Optionally, set normals
            if geometry.vertex_normals is not None:
                usd_mesh.GetNormalsAttr().Set([Gf.Vec3f(*normal) for normal in geometry.vertex_normals])
                usd_mesh.SetNormalsInterpolation("vertex")

            # Handle materials and other attributes if needed
    else:
        # It's a single mesh, proceed as before
        usd_mesh = UsdGeom.Mesh.Define(stage, "/Mesh")

        # Set vertex positions
        usd_mesh.GetPointsAttr().Set([Gf.Vec3f(*vertex) for vertex in mesh.vertices])

        # Set face indices and counts
        face_vertex_indices = mesh.faces.flatten().tolist()
        face_vertex_counts = [len(face) for face in mesh.faces]

        usd_mesh.GetFaceVertexIndicesAttr().Set(face_vertex_indices)
        usd_mesh.GetFaceVertexCountsAttr().Set(face_vertex_counts)

        # Optionally, set normals
        if mesh.vertex_normals is not None:
            usd_mesh.GetNormalsAttr().Set([Gf.Vec3f(*normal) for normal in mesh.vertex_normals])
            usd_mesh.SetNormalsInterpolation("vertex")

    # Save the USD file
    stage.GetRootLayer().Save()
    print(f"USD file successfully exported to {output_file}")


if __name__ == "__main__":
    input_file = "/workspace/Data/maisi_ct_generative/datasets/output_scene.gltf"  # or "input.obj"
    output_file = "/workspace/Code/tutorials/modules/omniverse/output_scene.usd"

    convert_mesh_to_usd(input_file, output_file)
