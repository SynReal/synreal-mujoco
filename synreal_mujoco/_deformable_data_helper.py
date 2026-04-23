import numpy as np

# load tetrahedron mesh from a VTK unstructured grid file (ASCII)
def load_tetrahedrons(file_path):
    vertices = []
    tets = []

    with open(file_path, 'r') as f:
        lines = [l.rstrip() for l in f]

    i = 0
    while i < len(lines):
        parts = lines[i].split()
        if not parts:
            i += 1
            continue

        if parts[0] == 'POINTS':
            n_points = int(parts[1])
            i += 1
            while len(vertices) < n_points:
                row = lines[i].split()
                i += 1
                if not row or row[0].startswith('#'):
                    continue
                # a POINTS line can pack multiple xyz triplets
                for j in range(0, len(row), 3):
                    vertices.append([float(row[j]), float(row[j+1]), float(row[j+2])])

        elif parts[0] == 'CELLS':
            n_cells = int(parts[1])
            i += 1
            for _ in range(n_cells):
                row = lines[i].split()
                n_verts = int(row[0])
                if n_verts == 4:
                    tets.append([int(row[1]), int(row[2]), int(row[3]), int(row[4])])
                i += 1

        else:
            i += 1

    return np.array(vertices, dtype=float), np.array(tets, dtype=int)

def compute_boundary_faces(tets: np.ndarray) -> np.ndarray:
    """Returns triangle faces on the boundary of a tet mesh (appear in exactly one tet)."""
    # Each tet has 4 triangular faces defined by combinations of its 4 vertices
    face_local = np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]])
    # Build all faces: shape (N*4, 3)
    all_faces = tets[:, face_local].reshape(-1, 3)
    # Canonical form: sort each face's vertex indices so shared faces compare equal
    sorted_faces = np.sort(all_faces, axis=1)
    # Find faces that appear exactly once (boundary = not shared)
    _, inverse, counts = np.unique(sorted_faces, axis=0, return_inverse=True, return_counts=True)
    boundary_mask = counts[inverse] == 1
    return all_faces[boundary_mask]
