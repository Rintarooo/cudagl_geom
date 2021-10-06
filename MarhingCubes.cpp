// #include <gl/glew.h>
// #include <gl/freeglut.h>
#include <GL/glew.h>
#include <GL/glut.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "Metaball.h"
#include "BasicDef.h"

// 等値面を定める濃度値．
float iso_value = 1.0f;

// メタボール法で定義された大域変数．
extern uint num_voxels;

// マーチングキューブ法で一時的に参照する大域変数．
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

// 外部定義の関数．
extern void thrustScan(unsigned int *input, unsigned int num_elements, unsigned int *output);
extern void launchClassifyVoxel(uint num_voxels, float iso_value, uint *voxel_vertices, uint *voxel_occupied);
extern void launchCompactVoxels(uint num_voxels, uint *voxel_occupied, uint *voxel_occupied_scanned, uint *compacted_voxel);
extern void launchGeneratePolys(uint num_occupied_voxels, uint max_num_vertices, float iso_value, 
	uint *compacted_voxel, uint *num_vertices_scanned, float4 *poly_pos, float4 *poly_normal);

// 頂点バッファオブジェクトの生成．
void createVBO(GLuint *vbo, unsigned int size, struct cudaGraphicsResource **vbo_res, unsigned int vbo_res_flags)
// GLuint *vbo; 頂点バッファオブジェクト．
// unsigned int size; バッファのサイズ．
// struct cudaGraphicsResource **vbo_res; CUDAのグラフィックスリソース．
// unsigned int vbo_res_flags; グラフィックスリソースの使い方についてのヒント．
{
    glGenBuffers(1, vbo);
    glBindBuffer(GL_ARRAY_BUFFER, *vbo);
	glBufferData(GL_ARRAY_BUFFER, size, NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // 生成した頂点バッファオブジェクトをグラフィックスリソースに登録．
	cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags);
}

// 頂点バッファオブジェクトの削除．
void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res)
// GLuint *vbo; 頂点バッファオブジェクト．
// struct cudaGraphicsResource *vbo_res; CUDAのグラフィックスリソース．
{

	// 頂点バッファオブジェクトを登録から外す．
	cudaGraphicsUnregisterResource(vbo_res);
    glDeleteBuffers(1, vbo);
    *vbo = 0;
}

// マーチングキューブ法による等値面のポリゴン化．
void marchingCubesIsoSurface(void)
{
    uint last_elem, last_scan_elem;
 
    // ポリゴンを貼り付けるセルの選択．
    launchClassifyVoxel(num_voxels, iso_value, d_voxel_vertices, d_voxel_occupied);

    // 分類後のセルをスキャンし個数を排他的に積算．
	thrustScan(d_voxel_occupied, num_voxels, d_voxel_occupied_scanned);

    // スキャン結果に基づいて処理対象のセル数を得る．スキャン結果は
    // この後のセルの圧縮でも用いる．排他的な積算なので，最後の個数を
    // 別途加える必要がある．
    cudaMemcpy((void *)&last_elem, (void *)(d_voxel_occupied + num_voxels - 1), 
        sizeof(uint), cudaMemcpyDeviceToHost);
    cudaMemcpy((void *)&last_scan_elem, (void *)(d_voxel_occupied_scanned + num_voxels - 1), 
        sizeof(uint), cudaMemcpyDeviceToHost);
    num_occupied_voxels = last_elem + last_scan_elem;

    // 処理すべきセルがゼロのケース．
    if (num_occupied_voxels == 0) {
        total_num_vertices = 0;
        return;
    }

    // 処理対象のセルの圧縮．
    launchCompactVoxels(num_voxels, d_voxel_occupied, d_voxel_occupied_scanned, d_compacted_voxel);

    // 各セルについてポリゴン貼り付けに必要な頂点数を排他的に積算．
	thrustScan(d_voxel_vertices, num_voxels, d_voxel_vertices_scanned);

    // 必要な総頂点数を得る．
    cudaMemcpy((void *)&last_elem, (void *)(d_voxel_vertices + num_voxels - 1), 
        sizeof(uint), cudaMemcpyDeviceToHost);
    cudaMemcpy((void *)&last_scan_elem, (void *)(d_voxel_vertices_scanned + num_voxels - 1),
        sizeof(uint), cudaMemcpyDeviceToHost);
    total_num_vertices = last_elem + last_scan_elem;

    // 頂点情報と法線情報のグラフィックスリソースをデバイス変数d_poly_posとd_poly_normalへマップ．
	cudaGraphicsMapResources(1, &poly_pos_vbo_res, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&d_poly_pos, NULL, poly_pos_vbo_res);
	cudaGraphicsMapResources(1, &poly_normal_vbo_res, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&d_poly_normal, NULL, poly_normal_vbo_res);

	// セル内部へのポリゴンの貼り付け．
    launchGeneratePolys(num_occupied_voxels, max_num_vertices, iso_value, d_compacted_voxel, d_voxel_vertices_scanned, 
        d_poly_pos, d_poly_normal);

	// アンマップ．
	cudaGraphicsUnmapResources(1, &poly_pos_vbo_res, 0);
	cudaGraphicsUnmapResources(1, &poly_normal_vbo_res, 0);
}

// 等値面を表すポリゴン群のVBOを用いた描画．
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

