# Creating the face file from the chosen submesh
import itertools
from submesh import idx_from_matrix_coord, submesh_idx_from_num_vertices_in_each_direction, matrix_coord_from_idx

def create_face_file_from_num_vertices(submesh_num_vertices_vertical, 
                                       submesh_num_vertices_horizontal,
                                       sequence_name = 'TowelWall',
                                       dataset_number = '2',
                                       num_vertices_height = 52,
                                       num_vertices_width = 103):
    submesh_idx = submesh_idx_from_num_vertices_in_each_direction(
        submesh_num_vertices_vertical = submesh_num_vertices_vertical,
        submesh_num_vertices_horizontal = submesh_num_vertices_horizontal)
#     print(submesh_idx)
    matrix_idx = [matrix_coord_from_idx(i) for i in submesh_idx]
#     print(matrix_idx)
    
    face_text = ''
    for vertical, horizontal in matrix_idx:
        if (vertical!=num_vertices_height-1 and horizontal!=num_vertices_width-1):
    #         print(vertical, horizontal)
            next_vertex_h = [horizontal_aux for vertical_aux, horizontal_aux in matrix_idx if (vertical_aux== vertical and horizontal_aux>horizontal)]
            next_vertex_h = min(next_vertex_h)
            next_vertex_v = [vertical_aux for vertical_aux, horizontal_aux in matrix_idx if (vertical_aux>vertical and horizontal_aux==horizontal)]
            next_vertex_v = min(next_vertex_v)
            
            # With the indexes of the submesh (i.e. in range(submesh_num_vertices_vertical*submesh_num_vertices_horizontal)
            face_text += str(submesh_idx.index(idx_from_matrix_coord(matrix_coord=(vertical, horizontal)))) + ' '
            face_text += str(submesh_idx.index(idx_from_matrix_coord(matrix_coord=(vertical, next_vertex_h)))) + ' '
            face_text += str(submesh_idx.index(idx_from_matrix_coord(matrix_coord=(next_vertex_v, horizontal)))) + ' '
            face_text += str(submesh_idx.index(idx_from_matrix_coord(matrix_coord=(next_vertex_v, next_vertex_h))))
            face_text += '\n'
            # With the indexes of the submesh (i.e. in range(num_vertices_height*num_vertices_width)
#             face_text += str(idx_from_matrix_coord(matrix_coord=(vertical, horizontal))) + ' '
#             face_text += str(idx_from_matrix_coord(matrix_coord=(vertical, horizontal+1))) + ' '
#             face_text += str(idx_from_matrix_coord(matrix_coord=(vertical+1, horizontal))) + ' '
#             face_text += str(idx_from_matrix_coord(matrix_coord=(vertical+1, horizontal+1)))
#             face_text += '\n'
    
#     print(face_text)
    filename = 'Renders' + sequence_name + dataset_number + '/faces_submesh_'
    filename += str(submesh_num_vertices_vertical) + 'by' + str(submesh_num_vertices_horizontal) + '.txt'
    with open(filename, 'w') as x_file:
        x_file.write(face_text)

if __name__ == '__main__':
#     submesh_num_vertices_vertical = 2
#     submesh_num_vertices_horizontal = 3
    submesh_num_vertices_vertical = 2
    submesh_num_vertices_horizontal = 10
    dataset_number = '11'
    create_face_file_from_num_vertices(submesh_num_vertices_vertical=submesh_num_vertices_vertical,
                                       submesh_num_vertices_horizontal=submesh_num_vertices_horizontal,
                                      dataset_number=dataset_number)