def center_obj_by_centroid(file_path):
    vertices = []
    normals = []
    texture_coords = []
    faces = []
    other_lines = []

    # Read the original OBJ file and collect vertices, normals, texture coordinates, and faces
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):  # Vertex data
                parts = line.split()
                x, y, z = map(float, parts[1:4])
                vertices.append((x, y, z))
            elif line.startswith('vn '):  # Normal data
                parts = line.split()
                nx, ny, nz = map(float, parts[1:4])
                normals.append((nx, ny, nz))
            elif line.startswith('vt '):  # Texture coordinate data
                parts = line.split()
                u, v = map(float, parts[1:3])
                texture_coords.append((u, v))
            elif line.startswith('f '):  # Face data
                faces.append(line)
            else:
                other_lines.append(line)

    # Calculate the centroid
    if vertices:
        centroid_x = sum(v[0] for v in vertices) / len(vertices)
        centroid_y = sum(v[1] for v in vertices) / len(vertices)
        centroid_z = sum(v[2] for v in vertices) / len(vertices)

        print(f"Centroid: ({centroid_x}, {centroid_y}, {centroid_z})")

        # Translate vertices so the centroid is at the origin
        translated_vertices = [(v[0] - centroid_x, v[1], v[2] - centroid_z) for v in vertices]

        # Prepare the content to write back
        lines_to_write = other_lines + \
                         [f"v {v[0]} {v[1]} {v[2]}\n" for v in translated_vertices] + \
                         [f"vn {n[0]} {n[1]} {n[2]}\n" for n in normals] + \
                         [f"vt {t[0]} {t[1]}\n" for t in texture_coords] + \
                         faces

        # Write the modified data to a new OBJ file
        with open(file_path.replace('.obj', '_centered.obj'), 'w') as file:
            file.writelines(lines_to_write)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python center_obj.py <path_to_obj_file>")
        sys.exit(1)
    file_path = sys.argv[1]
    center_obj_by_centroid(file_path)
