// #include <gl/glew.h>
// #include <gl/freeglut.h>
#include <GL/glew.h>
#include <GL/glut.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "Metaball.h"
#include "BasicDef.h"

// ���l�ʂ��߂�Z�x�l�D
float iso_value = 1.0f;

// ���^�{�[���@�Œ�`���ꂽ���ϐ��D
extern uint num_voxels;

// �}�[�`���O�L���[�u�@�ňꎞ�I�ɎQ�Ƃ�����ϐ��D
uint max_num_vertices;
uint num_occupied_voxels;
uint total_num_vertices;

uint *d_num_vertices_table;
uint *d_triangle_table;
uint *d_voxel_vertices;
uint *d_voxel_vertices_scanned;
uint *d_voxel_occupied;
uint *d_voxel_occupied_scanned;
uint *d_compacted_voxel;

GLuint poly_pos_vbo, poly_normal_vbo;
struct cudaGraphicsResource *poly_pos_vbo_res, *poly_normal_vbo_res;
float4 *d_poly_pos, *d_poly_normal;

// �O����`�̊֐��D
extern void thrustScan(unsigned int *input, unsigned int num_elements, unsigned int *output);
extern void launchClassifyVoxel(uint num_voxels, float iso_value, uint *voxel_vertices, uint *voxel_occupied);
extern void launchCompactVoxels(uint num_voxels, uint *voxel_occupied, uint *voxel_occupied_scanned, uint *compacted_voxel);
extern void launchGeneratePolys(uint num_occupied_voxels, uint max_num_vertices, float iso_value, 
	uint *compacted_voxel, uint *num_vertices_scanned, float4 *poly_pos, float4 *poly_normal);

// ���_�o�b�t�@�I�u�W�F�N�g�̐����D
void createVBO(GLuint *vbo, unsigned int size, struct cudaGraphicsResource **vbo_res, unsigned int vbo_res_flags)
// GLuint *vbo; ���_�o�b�t�@�I�u�W�F�N�g�D
// unsigned int size; �o�b�t�@�̃T�C�Y�D
// struct cudaGraphicsResource **vbo_res; CUDA�̃O���t�B�b�N�X���\�[�X�D
// unsigned int vbo_res_flags; �O���t�B�b�N�X���\�[�X�̎g�����ɂ��Ẵq���g�D
{
    glGenBuffers(1, vbo);
    glBindBuffer(GL_ARRAY_BUFFER, *vbo);
	glBufferData(GL_ARRAY_BUFFER, size, NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // �����������_�o�b�t�@�I�u�W�F�N�g���O���t�B�b�N�X���\�[�X�ɓo�^�D
	cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags);
}

// ���_�o�b�t�@�I�u�W�F�N�g�̍폜�D
void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res)
// GLuint *vbo; ���_�o�b�t�@�I�u�W�F�N�g�D
// struct cudaGraphicsResource *vbo_res; CUDA�̃O���t�B�b�N�X���\�[�X�D
{

	// ���_�o�b�t�@�I�u�W�F�N�g��o�^����O���D
	cudaGraphicsUnregisterResource(vbo_res);
    glDeleteBuffers(1, vbo);
    *vbo = 0;
}

// �}�[�`���O�L���[�u�@�ɂ�铙�l�ʂ̃|���S�����D
void marchingCubesIsoSurface(void)
{
    uint last_elem, last_scan_elem;
 
    // �|���S����\��t����Z���̑I���D
    launchClassifyVoxel(num_voxels, iso_value, d_voxel_vertices, d_voxel_occupied);

    // ���ތ�̃Z�����X�L����������r���I�ɐώZ�D
	thrustScan(d_voxel_occupied, num_voxels, d_voxel_occupied_scanned);

    // �X�L�������ʂɊ�Â��ď����Ώۂ̃Z�����𓾂�D�X�L�������ʂ�
    // ���̌�̃Z���̈��k�ł��p����D�r���I�ȐώZ�Ȃ̂ŁC�Ō�̌���
    // �ʓr������K�v������D
    cudaMemcpy((void *)&last_elem, (void *)(d_voxel_occupied + num_voxels - 1), 
        sizeof(uint), cudaMemcpyDeviceToHost);
    cudaMemcpy((void *)&last_scan_elem, (void *)(d_voxel_occupied_scanned + num_voxels - 1), 
        sizeof(uint), cudaMemcpyDeviceToHost);
    num_occupied_voxels = last_elem + last_scan_elem;

    // �������ׂ��Z�����[���̃P�[�X�D
    if (num_occupied_voxels == 0) {
        total_num_vertices = 0;
        return;
    }

    // �����Ώۂ̃Z���̈��k�D
    launchCompactVoxels(num_voxels, d_voxel_occupied, d_voxel_occupied_scanned, d_compacted_voxel);

    // �e�Z���ɂ��ă|���S���\��t���ɕK�v�Ȓ��_����r���I�ɐώZ�D
	thrustScan(d_voxel_vertices, num_voxels, d_voxel_vertices_scanned);

    // �K�v�ȑ����_���𓾂�D
    cudaMemcpy((void *)&last_elem, (void *)(d_voxel_vertices + num_voxels - 1), 
        sizeof(uint), cudaMemcpyDeviceToHost);
    cudaMemcpy((void *)&last_scan_elem, (void *)(d_voxel_vertices_scanned + num_voxels - 1),
        sizeof(uint), cudaMemcpyDeviceToHost);
    total_num_vertices = last_elem + last_scan_elem;

    // ���_���Ɩ@�����̃O���t�B�b�N�X���\�[�X���f�o�C�X�ϐ�d_poly_pos��d_poly_normal�փ}�b�v�D
	cudaGraphicsMapResources(1, &poly_pos_vbo_res, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&d_poly_pos, NULL, poly_pos_vbo_res);
	cudaGraphicsMapResources(1, &poly_normal_vbo_res, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&d_poly_normal, NULL, poly_normal_vbo_res);

	// �Z�������ւ̃|���S���̓\��t���D
    launchGeneratePolys(num_occupied_voxels, max_num_vertices, iso_value, d_compacted_voxel, d_voxel_vertices_scanned, 
        d_poly_pos, d_poly_normal);

	// �A���}�b�v�D
	cudaGraphicsUnmapResources(1, &poly_pos_vbo_res, 0);
	cudaGraphicsUnmapResources(1, &poly_normal_vbo_res, 0);
}

// ���l�ʂ�\���|���S���Q��VBO��p�����`��D
void drawIsoSurface(void)
{
	glBindBuffer(GL_ARRAY_BUFFER, poly_pos_vbo);
	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(4, GL_FLOAT, 0, 0);
	
	glBindBuffer(GL_ARRAY_BUFFER, poly_normal_vbo);
	glEnableClientState(GL_NORMAL_ARRAY);
	glNormalPointer(GL_FLOAT, 4 * sizeof(float), 0);

	glDrawArrays(GL_TRIANGLES, 0, total_num_vertices);

	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_NORMAL_ARRAY);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

