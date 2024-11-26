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


def convert_to_mesh(segmentation_path, output_folder, filename, label_value=1, smoothing_factor=0.5, reduction_ratio=0.5):
    """
    Function to perform segmentation-to-mesh conversion and smoothing
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Step 1: Load segmentation (binary labelmap, e.g., NRRD file)
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(segmentation_path)
    reader.Update()

    label_values = [label_value] if isinstance(label_value, int) else label_value
    if len(label_values) > 1:
        append_filter = vtk.vtkAppendPolyData()  # To combine all labels' surfaces
    for i in label_values:
        # Step 2: Create Closed Surface Representation using vtkDiscreteFlyingEdges3D
        flying_edges = vtk.vtkDiscreteFlyingEdges3D()
        flying_edges.SetInputConnection(reader.GetOutputPort())
        flying_edges.ComputeGradientsOff()
        flying_edges.ComputeNormalsOff()
        flying_edges.SetValue(0, i)  # Assuming label 1 for segmentation surface
        flying_edges.Update()

        if reduction_ratio is not None:
            decimation_filter = vtk.vtkDecimatePro()
            decimation_filter.SetInputConnection(flying_edges.GetOutputPort())
            decimation_filter.SetFeatureAngle(60)
            decimation_filter.SplittingOff()
            decimation_filter.PreserveTopologyOn()
            decimation_filter.SetMaximumError(1)
            decimation_filter.SetTargetReduction(reduction_ratio)  # Adjust reduction level (0.0 to 1.0)
            decimation_filter.Update()

        # Step 3: Smooth the resulting mesh
        smoothing_filter = vtk.vtkWindowedSincPolyDataFilter()
        numberOfIterations = int(20 + smoothing_factor * 40)
        passBand = pow(10.0, -4.0 * smoothing_factor)
        if reduction_ratio is not None:
            smoothing_filter.SetInputConnection(decimation_filter.GetOutputPort())
        else:
            smoothing_filter.SetInputConnection(flying_edges.GetOutputPort())
        smoothing_filter.SetNumberOfIterations(numberOfIterations)  # Smooth iterations
        smoothing_filter.SetPassBand(passBand)  # Smoothing passband
        smoothing_filter.FeatureEdgeSmoothingOff()
        smoothing_filter.NonManifoldSmoothingOn()
        smoothing_filter.NormalizeCoordinatesOn()
        smoothing_filter.Update()

        # Step 4: Generate normals for better shading
        normals_filter = vtk.vtkPolyDataNormals()
        normals_filter.SetInputConnection(smoothing_filter.GetOutputPort())
        normals_filter.ConsistencyOn()
        normals_filter.SplittingOff()
        normals_filter.Update()
        
        if len(label_values) > 1:
            append_filter.AddInputData(normals_filter.GetOutput())
            
    if len(label_values) > 1:
        append_filter.Update()
        polydata = append_filter.GetOutput()
        print("Number of points in PolyData:", polydata.GetNumberOfPoints())
        print("Number of cells in PolyData:", polydata.GetNumberOfCells())
    else:
        polydata = normals_filter.GetOutput()
    # Step 5: Export the smoothed mesh to glTF
    writer = vtk.vtkOBJWriter()
    output_filename = os.path.join(output_folder, filename)
    writer.SetFileName(output_filename)
    writer.SetInputData(polydata)  # Use the polydata object
    writer.Write()

    print(f"Mesh successfully exported to {output_filename}")


def convert_obj_to_usd(obj_filename, usd_filename):
    """
    Function to convert an OBJ file to USD
    """
    # Create a new USD stage
    stage = Usd.Stage.CreateNew(usd_filename)

    # Define a mesh at the root of the stage
    mesh = UsdGeom.Mesh.Define(stage, '/RootMesh')

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
    with open(obj_filename, 'r') as obj_file:
        for line in obj_file:
            if line.startswith('v '):
                # Vertex position
                _, x, y, z = line.strip().split()
                vertices.append((float(x), float(y), float(z)))
            elif line.startswith('vn '):
                # Vertex normal
                _, nx, ny, nz = line.strip().split()
                normals.append((float(nx), float(ny), float(nz)))
            elif line.startswith('vt '):
                # Texture coordinate
                _, u, v = line.strip().split()
                texcoords.append((float(u), float(v)))
            elif line.startswith('f '):
                # Face
                face_elements = line.strip().split()[1:]
                vertex_count = len(face_elements)
                face_vertex_counts.append(vertex_count)
                for elem in face_elements:
                    indices = elem.split('/')
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
        mesh.SetNormalsInterpolation('faceVarying')  # Adjust based on how normals are specified

    # Optionally set texture coordinates if they exist
    if texcoords and texcoord_indices:
        # Reorder texcoords according to face vertices
        ordered_texcoords = [texcoords[i] for i in texcoord_indices]
        stPrimvar = mesh.CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
        stPrimvar.Set([Gf.Vec2f(*tc) for tc in ordered_texcoords])

    # Save the stage
    stage.GetRootLayer().Save()
    
    print(f"USD file successfully exported to {usd_filename}")

