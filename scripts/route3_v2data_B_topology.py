"""Audit opaque open and closed meshes for unambiguous depth ordering."""
from collections import deque

import bmesh
from mathutils.bvhtree import BVHTree
import numpy as np

from route3_v2data_B_mesh_check import Rejected


PLANE_TOLERANCE = 2e-7


def face_fans(mesh):
    """Check single fans at vertices and consistent traversal of interior edges."""
    for edge in mesh.edges:
        count = len(edge.link_faces)
        if count not in {1, 2} or (count == 2 and not edge.is_contiguous):
            raise Rejected("edge has an ambiguous face incidence or inconsistent winding")
    for vertex in mesh.verts:
        faces = set(vertex.link_faces)
        if not faces or sum(edge.is_boundary for edge in vertex.link_edges) not in {0, 2}:
            raise Rejected("vertex has an ambiguous boundary fan")
        pending = [faces.pop()]
        while pending:
            current = pending.pop()
            adjacent = {face for edge in current.edges if vertex in edge.verts for face in edge.link_faces}
            unseen = faces.intersection(adjacent)
            faces.difference_update(unseen)
            pending.extend(unseen)
        if faces:
            raise Rejected("vertex has disconnected face fans")


def check_closed_winding(mesh):
    remaining = set(mesh.faces)
    while remaining:
        first = remaining.pop()
        component = [first]
        pending = deque([first])
        while pending:
            face = pending.pop()
            adjacent = {neighbor for edge in face.edges for neighbor in edge.link_faces}
            unseen = remaining.intersection(adjacent)
            remaining.difference_update(unseen)
            component.extend(unseen)
            pending.extend(unseen)
        if any(edge.is_boundary for face in component for edge in face.edges):
            continue
        origin = component[0].verts[0].co
        volume = sum((a.co-origin).dot((b.co-origin).cross(c.co-origin))
                     for a,b,c in (face.verts for face in component))
        if volume <= 0:
            raise Rejected("closed component is inward facing or has zero signed volume")


def interval_on_intersection(triangle, plane_distances, direction):
    projected = [float(point @ direction) for point, distance in zip(triangle,plane_distances)
                 if abs(distance) <= PLANE_TOLERANCE]
    for i,j in ((0,1),(1,2),(2,0)):
        a,b = plane_distances[i],plane_distances[j]
        if (a < -PLANE_TOLERANCE and b > PLANE_TOLERANCE
                or b < -PLANE_TOLERANCE and a > PLANE_TOLERANCE):
            point = triangle[i] + (triangle[j]-triangle[i]) * a/(a-b)
            projected.append(float(point @ direction))
    return (min(projected),max(projected)) if projected else None


def interior_overlap(a, b, normal):
    axes = [axis for axis in range(3) if axis != abs(normal).argmax()]
    a,b = a[:,axes],b[:,axes]
    edges = np.concatenate((np.roll(a,-1,axis=0)-a, np.roll(b,-1,axis=0)-b))
    perpendicular = edges[:,[1,0]] * [1,-1]
    tolerance = PLANE_TOLERANCE*np.linalg.norm(perpendicular,axis=1)
    x,y = a @ perpendicular.T,b @ perpendicular.T
    overlap = np.minimum(x.max(0),y.max(0))-np.maximum(x.min(0),y.min(0))
    return bool(np.all(overlap > tolerance))


def check_crossing(vertices, indices):
    triangles = vertices[indices]
    normals = np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    normals /= np.linalg.norm(normals,axis=1)[:,None]
    tree = BVHTree.FromPolygons(vertices.tolist(),indices.tolist(),all_triangles=True)
    for i,j in tree.overlap(tree):
        if i >= j:
            continue
        shared = len(set(indices[i]).intersection(indices[j]))
        if shared >= 2:
            continue
        a,b = triangles[i],triangles[j]
        d1,d2 = (a-b[0]) @ normals[j],(b-a[0]) @ normals[i]
        direction = np.cross(normals[i],normals[j])
        length = np.linalg.norm(direction)
        if length <= PLANE_TOLERANCE:
            continue
        direction /= length
        x = interval_on_intersection(a,d1,direction)
        y = interval_on_intersection(b,d2,direction)
        if x is None or y is None:
            continue
        overlap = min(x[1],y[1])-max(x[0],y[0])
        if overlap > PLANE_TOLERANCE or (shared == 0 and overlap >= -PLANE_TOLERANCE):
            raise Rejected("non-adjacent triangles intersect")


def check_coplanar_overlap(vertices, indices):
    triangles = vertices[indices]
    directions = np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    magnitude = np.linalg.norm(directions,axis=1)
    if np.any(magnitude == 0):
        raise Rejected("mesh contains a zero-area triangle")
    normals = directions/magnitude[:,None]
    # Blender's triangle overlap query does not return coplanar triangle pairs.
    lower,upper = triangles.min(1),triangles.max(1)
    axis = int(np.ptp(vertices,axis=0).argmax())
    order = np.argsort(lower[:,axis],kind="stable")
    starts = lower[order,axis]
    for position,i in enumerate(order):
        last = np.searchsorted(starts,upper[i,axis]+PLANE_TOLERANCE,side="right")
        nearby = order[position+1:last]
        fits = ((lower[nearby] <= upper[i]+PLANE_TOLERANCE) &
                (upper[nearby] >= lower[i]-PLANE_TOLERANCE)).all(1)
        nearby = nearby[fits]
        coplanar = (np.abs((triangles[nearby]-triangles[i,0]) @ normals[i])
                    <= PLANE_TOLERANCE).all(1)
        for j in nearby[coplanar]:
            if interior_overlap(triangles[i],triangles[j],normals[i]):
                raise Rejected("coplanar triangle interiors overlap")


def audit_mesh(ob):
    geometry = ob.data
    geometry.calc_loop_triangles()
    raw = np.asarray([ob.matrix_world @ v.co for v in geometry.vertices],dtype=np.float64)
    if not len(raw) or not np.isfinite(raw).all():
        raise Rejected("mesh has empty or nonfinite coordinates")
    extent = float(np.ptp(raw,axis=0).max())
    if not .0005 <= extent <= 500:
        raise Rejected("mesh has invalid scale")
    indices = np.asarray([list(t.vertices) for t in geometry.loop_triangles],dtype=np.int64)
    if not len(indices):
        raise Rejected("mesh has no faces")
    check_coplanar_overlap((raw-raw.min(0))/extent,indices)
    mesh = bmesh.new()
    try:
        mesh.from_mesh(ob.data)
        if not mesh.faces:
            raise Rejected("mesh has no faces")
        bmesh.ops.transform(mesh,matrix=ob.matrix_world,verts=list(mesh.verts))
        bmesh.ops.remove_doubles(mesh,verts=list(mesh.verts),dist=max(1e-8,extent*1e-6))
        bmesh.ops.triangulate(mesh,faces=list(mesh.faces))
        if not mesh.faces or any(face.calc_area() <= 1e-14 for face in mesh.faces):
            raise Rejected("mesh contains a zero-area triangle")
        face_fans(mesh)
        check_closed_winding(mesh)
        mesh.verts.index_update()
        points = np.asarray([v.co for v in mesh.verts],dtype=np.float64)
        points = (points-points.min(0))/extent
        triangles = np.asarray([[v.index for v in face.verts] for face in mesh.faces])
        check_crossing(points,triangles)
        return any(edge.is_boundary for edge in mesh.edges)
    finally:
        mesh.free()
