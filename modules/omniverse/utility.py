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
import json
from pxr import Usd, UsdGeom, Gf, Sdf, UsdShade
import trimesh
import numpy as np
import matplotlib.pyplot as plt
import random
import colorsys


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

    nifti_transform_matrix = reader.GetSFormMatrix()
    if nifti_transform_matrix is None or nifti_transform_matrix.IsIdentity():
        nifti_transform_matrix = vtk.vtkMatrix4x4()

    nifti_transform = vtk.vtkTransform()
    nifti_transform.SetMatrix(nifti_transform_matrix)

    label_values = {label_value: None} if isinstance(label_value, int) else label_value
    if len(label_values.keys()) > 1:
        renderer = vtk.vtkRenderer()
        render_window = vtk.vtkRenderWindow()
        render_window.AddRenderer(renderer)
        actor_metadata = {}
    for i, name in label_values.items():
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

        # Step 5: Decimate the mesh further
        decimation = vtk.vtkQuadricDecimation()
        decimation.SetInputConnection(smoothing_filter.GetOutputPort())
        decimation.SetTargetReduction(0.9)  # 90% reduction, the same as slicer
        decimation.VolumePreservationOn()
        decimation.Update()

        # Step 6: Generate normals for better shading
        decimatedNormals = vtk.vtkPolyDataNormals()
        decimatedNormals.SetInputConnection(decimation.GetOutputPort())
        decimatedNormals.SplittingOff()
        decimatedNormals.ConsistencyOn()
        decimatedNormals.Update()

        # Step 7: Apply NIFTI SForm transform
        nifti_transformer = vtk.vtkTransformPolyDataFilter()
        nifti_transformer.SetTransform(nifti_transform)
        nifti_transformer.SetInputConnection(decimatedNormals.GetOutputPort())
        nifti_transformer.Update()

        # Step 8: convert to LPS (apply after NIFTI transform)
        ras2lps = vtk.vtkMatrix4x4()
        ras2lps.SetElement(0, 0, -1)
        ras2lps.SetElement(1, 1, -1)
        ras2lpsTransform = vtk.vtkTransform()
        ras2lpsTransform.SetMatrix(ras2lps)
        transformer = vtk.vtkTransformPolyDataFilter()
        transformer.SetTransform(ras2lpsTransform)
        transformer.SetInputConnection(nifti_transformer.GetOutputPort())
        transformer.Update()

        if len(label_values.keys()) > 1:
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
            actor_metadata[actor] = name
            renderer.AddActor(actor)

    output_filename = os.path.join(output_folder, filename)
    if len(label_values.keys()) > 1:
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

    if len(label_values.keys()) > 1:
        # Modify GLTF to include actor names
        with open(output_filename, "r") as f:
            gltf_data = json.load(f)

        # Iterate over actors and add names to GLTF nodes
        actors = renderer.GetActors()
        actors.InitTraversal()

        for i, node in enumerate(gltf_data.get("nodes", [])):
            actor = actors.GetNextActor()
            if actor in actor_metadata:
                node["name"] = actor_metadata[actor]

        # Save the modified GLTF file
        modified_output_filename = output_filename.replace(".gltf", "_modified.gltf")
        with open(modified_output_filename, "w") as f:
            json.dump(gltf_data, f, indent=2)
        print(f"Modified GLTF successfully exported to {modified_output_filename}")


def convert_mesh_to_usd(input_file, output_file):
    """
    convert a mesh file to USD format
    """
    # Load the mesh
    mesh = trimesh.load(input_file)

    # Create a new USD stage
    stage = Usd.Stage.CreateNew(output_file)
    root = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(root.GetPrim())
    materials_path = "/World/Materials"
    UsdGeom.Scope.Define(stage, materials_path)

    # If the mesh is a Scene, process each geometry
    if isinstance(mesh, trimesh.Scene):
        for node_name in mesh.graph.nodes:
            if node_name == "world":
                continue
            geom_name = mesh.graph.get(node_name)[1]
            if geom_name is not None and geom_name.startswith("mesh"):
                print(f"Processing mesh: {node_name} {geom_name}")
                # Create a unique path for each mesh
                node_path = f"/World/{node_name}"
                xform = UsdGeom.Xform.Define(stage, node_path)
                # Define the Mesh under the Xform
                mesh_path = f"{node_path}/Mesh"
                usd_mesh = UsdGeom.Mesh.Define(stage, mesh_path)
                # get the geometry of the node
                geometry = mesh.geometry[geom_name]

                # Create a random color for this mesh
                # Using HSV for better color distribution
                h = random.random()  # Random hue
                s = 0.7 + 0.3 * random.random()  # Saturation between 0.7-1.0
                v = 0.7 + 0.3 * random.random()  # Value between 0.7-1.0
                r, g, b = colorsys.hsv_to_rgb(h, s, v)

                # Create a material with the random color
                mat_name = f"{node_name}_material"
                mat_path = f"{materials_path}/{mat_name}"
                material = UsdShade.Material.Define(stage, mat_path)

                # Create shader
                shader = UsdShade.Shader.Define(stage, f"{mat_path}/PreviewSurface")
                shader.CreateIdAttr("UsdPreviewSurface")

                # Set the random color
                shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(r, g, b))
                shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)

                # Connect shader to material
                material_output = material.CreateOutput("surface", Sdf.ValueTypeNames.Token)
                shader_output = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
                material_output.ConnectToSource(shader_output)

                # Bind material to mesh
                UsdShade.MaterialBindingAPI(usd_mesh).Bind(material)

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
