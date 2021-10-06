#include <cuda_runtime.h>

#include <math.h>
// #include "vector_types.h"// cuda vector types, such as float3, float4
#include "cuda_runtime.h"
#include <helper_math.h>


#include "Metaball.h"
#include "tables.h"

// thrust用のヘッダーファイル．
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>

// 格子構造のパラメータを保持した構造体．
__constant__ struct sim_params d_params;

// パラメータ設定．
void setParameters(struct sim_params *h_params)
{
    cudaMemcpyToSymbol(d_params, h_params, sizeof(struct sim_params), 0, cudaMemcpyHostToDevice);
}

// 1次元グリッドサイズの簡易計算．
void computeGridSize(uint n, uint block_size, uint *grid_size, uint *new_block_size)
// unit n; 処理対象の総数．
// unit block_size; 1ブロックあたりのスレッド数．
// uint &grid_size; グリッドサイズ．
// uint &new_block_size; 補正された1ブロックあたりのスレッド数．
{
    *new_block_size = min(block_size, n);
    *grid_size = ((n % *new_block_size == 0) ? (n / *new_block_size) : (n / *new_block_size + 1));
}

// 座標の3次元格子におけるグリッド位置．
__device__ int3 d_pointToGridPos(float4 point)
// float4 point; 座標値．
{
    int3 grid_pos;
    grid_pos.x = (int)floor((point.x - d_params.world_origin.x) / d_params.voxel_size.x);
    grid_pos.y = (int)floor((point.y - d_params.world_origin.y) / d_params.voxel_size.y);
    grid_pos.z = (int)floor((point.z - d_params.world_origin.z) / d_params.voxel_size.z);
    return grid_pos;
}

// グリッド位置におけるおける座標．
__device__ float4 d_gridPosToPoint(int3 grid_pos)
// int3 grid_pos; グリッド位置．
{
    float4 p;
    p.x = d_params.world_origin.x + grid_pos.x * d_params.voxel_size.x;
    p.y = d_params.world_origin.y + grid_pos.y * d_params.voxel_size.y;
    p.z = d_params.world_origin.z + grid_pos.z * d_params.voxel_size.z;
    p.w = 1.0f;
    return p;
}    

// 3次元的なグリッド位置から1次元的なセルアドレスへの変換．
__device__ uint d_gridPosToAddress(int3 grid_pos)
// int3 grid_pos; グリッド位置．
{

    // 位置が格子の範囲内に収まるように補正．
    if (grid_pos.x < 0) grid_pos.x = 0;
    if (grid_pos.x > (d_params.grid_size.x - 1)) grid_pos.x = (d_params.grid_size.x - 1);
    if (grid_pos.y < 0) grid_pos.y = 0;
    if (grid_pos.y > (d_params.grid_size.y - 1)) grid_pos.y = (d_params.grid_size.y - 1);
    if (grid_pos.z < 0) grid_pos.z = 0;
    if (grid_pos.z > (d_params.grid_size.z - 1)) grid_pos.z = (d_params.grid_size.z - 1);

    // 1次元的なセルアドレスの計算．
    return __umul24(__umul24(d_params.grid_size.x, d_params.grid_size.y), grid_pos.z) + __umul24(d_params.grid_size.x, grid_pos.y) + grid_pos.x;
}

// 3次元的なグリッド位置が格子の範囲内かどうかの判定．
__device__ bool d_validGridPosP(int3 grid_pos)
// int3 grid_pos; グリッド位置．
{
	return((grid_pos.x >= 0) && (grid_pos.x < d_params.grid_size.x) 
		&& (grid_pos.y >= 0) && (grid_pos.y < d_params.grid_size.y)
		&& (grid_pos.z >= 0) && (grid_pos.z < d_params.grid_size.z));
}

// セルアドレスからグリッド位置への変換．
__device__ int3 d_addressToGridPos(uint address)
// uint address; セルアドレス．
{
	uint tmp;
    int3 grid_pos;
    grid_pos.z = address / (d_params.grid_size.x * d_params.grid_size.y);
    tmp = address % (d_params.grid_size.x * d_params.grid_size.y);
    grid_pos.y = tmp / d_params.grid_size.x;
    grid_pos.x = tmp % d_params.grid_size.x; 
    return grid_pos;
}

// 座標へのセルアドレスの割り当て．
__global__ void d_calcAddress(uint num_points, float4 *point, uint *grid_point_address, uint *grid_point_index)
// uint num_points; 点の総数．
// float4 *point; 点群．
// unit *grid_point_address; 各点の3次元格子におけるセルアドレス．
// unit *grid_point_index; 各点のインデックス．
{
    uint index, address;

    // 処理対象の点の選択．
	index = __umul24(blockDim.x, blockIdx.x) + threadIdx.x;
    if (index >= num_points) 
        return;

    // セルアドレスの取得．
	address = d_gridPosToAddress(d_pointToGridPos(point[index]));

    // 得られたアドレスを点のインデックスとともに格納．
    grid_point_address[index] = address;
    grid_point_index[index] = index;
}

// 呼び出し関数．
void launchCalcAddress(uint num_points, float4 *point, uint *grid_point_address, uint *grid_point_index)
// uint num_points; 点の総数．
// float4 *point; 点群．
// unit *grid_point_address; 各点の3次元格子におけるセルアドレス．
// unit *grid_point_index; 各点のインデックス．
{
    uint grid, block;
    computeGridSize(num_points, 256, &grid, &block);
    d_calcAddress <<< grid, block >>> (num_points, point, grid_point_address, grid_point_index);
}

// キーに基づくソート．
void thrustSortByKey(unsigned int *key, unsigned int *values, unsigned int num_elements)
{
    thrust::sort_by_key(thrust::device_ptr<unsigned int>(key), 
		thrust::device_ptr<unsigned int>(key + num_elements),
		thrust::device_ptr<unsigned int>(values));
}

// 各格子セルに属する点の部分列の取得．
__global__ void d_reorderPointsAndAssignPointsToCells(uint num_points, float4 *point, 
	uint *grid_point_address, uint *grid_point_index,
    float4 *sorted_point, uint *cell_start, uint *cell_end)
// uint num_points; 点の総数．
// float4 *point; 点群．
// unit *grid_point_address; 各点のセルアドレス．アドレスに従ってソート済み．
// unit *grid_point_index; 各点のインデックス．アドレスに従ってソート済み．
// float4 *sorted_point; ソート済みの点群．
// unit *cell_start, *cell_end; 各格子セルに属するsorted_pointの部分列の範囲．
{
	extern __shared__ uint s_address[]; // 共有メモリはブロックサイズ+1の大きさに外部定義されている．
    uint index, address;

    // 処理対象の点の選択．
	index = __umul24(blockDim.x, blockIdx.x) + threadIdx.x;
    if (index >= num_points) 
        return;

	// 隣のスレッドでこの点のアドレスが参照できるように，一つずらして共有
	// メモリに格納．
	address = grid_point_address[index];
	s_address[threadIdx.x + 1] = address;
	if ((index > 0) && (threadIdx.x == 0)) {
		    
		// 共有メモリの先頭には一つ前のブロックの最後のアドレスを格納．
		s_address[0] = grid_point_address[index - 1];
	}
	__syncthreads();	

	// 一つ前の点とこの点とでアドレスの切り替わりがあった．
	if ((index == 0) || (address != s_address[threadIdx.x])) {
		cell_start[address] = index;
		if (index > 0)
			 cell_end[s_address[threadIdx.x]] = index;
	}
	if (index == (num_points - 1))
		cell_end[address] = num_points;

	// 点もソート順にしまい直しておく．
	sorted_point[index] = point[grid_point_index[index]];
}

// 呼び出し関数．
void launchReorderPointsAndAssignPointsToCells(uint num_points, uint num_voxels, float4 *point,
	uint *grid_point_address, uint *grid_point_index,
	float4 *sorted_point, uint *cell_start, uint *cell_end)
// uint num_points; 点の総数．
// uint num_voxels; セルの総数．
// float4 *point; 点群．
// unit *grid_point_address; 各点のセルアドレス．アドレスに従ってソート済み．
// unit *grid_point_index; 各点のインデックス．アドレスに従ってソート済み．
// float4 *sorted_point; ソート済みの点群．
// unit *cell_start, *cell_end; 各格子セルに属するsorted_pointの部分列の範囲．
{
    uint grid, block;

    // 予め全てのcell_startに0xffffffffをしまっておく．
	cudaMemset(cell_start, 0xffffffff, num_voxels * sizeof(uint));

    // 共有メモリサイズの外部定義．
	computeGridSize(num_points, 256, &grid, &block);  
	uint s_mem_size = (block + 1) * sizeof(uint);
    d_reorderPointsAndAssignPointsToCells <<< grid, block, s_mem_size >>> (num_points,
        point, grid_point_address, grid_point_index, sorted_point, cell_start, cell_end);
}

// 格子点についてセル中の粒子の濃度を積算．
__device__ float d_accumulatePointsInCell(float4 grid_point, int3 grid_pos, float4* sorted_point, 
	uint *cell_start, uint *cell_end, float limit_dist)
// float4 grid_point; 格子上の点．
// int3 grid_pos; セルのグリッド位置．このセル中の点の影響を積算．
// float4 *sorted_point; ソート済みの点群．
// unit *cell_start, *cell_end; 各格子セルに属するsorted_pointの部分列の範囲．
// float limit_dist; 打ち切り距離．
{
	uint i;
    uint address = d_gridPosToAddress(grid_pos);
	float value = 0.0f, d2;
    float4 rel_pos;

    // セルに点が含まれている．
    if (d_validGridPosP(grid_pos) && (cell_start[address] != 0xffffffff)) {

        // このセル中の点の影響を積算．
		for(i = cell_start[address]; i < cell_end[address]; i++) {
 
            // 点と格子上の点の相対位置を計算．
            rel_pos = sorted_point[i] - grid_point;
            d2 = rel_pos.x * rel_pos.x + rel_pos.y * rel_pos.y + rel_pos.z * rel_pos.z;
            if (d2 < limit_dist * limit_dist) // 打ち切り距離内であれば積算．
                value += 0.0002f / d2;
        }
    }
    return value;
}

// 各格子点への濃度値の割り当て．
__global__ void d_setDensityToGrid(uint num_voxels, float4 *sorted_point, uint *cell_start, uint *cell_end, float limit_dist, float *d_density)
// uint num_voxels; セルの総数．
// float4 *sorted_point; ソート済みの点群．
// unit *cell_start, *cell_end; 各格子セルに属するsorted_pointの部分列の範囲．
// float limit_dist; 打ち切り距離．
// float *d_density; 各格子の濃度値．
{
    uint block_index = __umul24(gridDim.x, blockIdx.y) + blockIdx.x; 
    uint address = __umul24(blockDim.x, block_index) + threadIdx.x;
	int x, y, z;
    int3 near_cell, grid_range;
	int3 grid_pos;
	float value;
	float4 grid_point;

    // このスレッドのインデックス（＝セルアドレス）はセルの総数をオーバーしている．
    if (address >= num_voxels) 
        return;

	// チェックするセルの範囲．1が望ましい．
	grid_range.x = (int)ceil(limit_dist / d_params.voxel_size.x);
	grid_range.y = (int)ceil(limit_dist / d_params.voxel_size.y);
	grid_range.z = (int)ceil(limit_dist / d_params.voxel_size.z);

    // 格子点のグリッド位置を得る．
    grid_pos = d_addressToGridPos(address);

    // 格子の座標の取得．
    grid_point = d_gridPosToPoint(grid_pos);

    // 近傍のセル中の点による濃度値を積算．
    value = 0.0f;
    for (z = - grid_range.z; z < grid_range.z; z++) {
        for (y = - grid_range.y; y < grid_range.y; y++) {
            for (x = - grid_range.x; x < grid_range.x; x++) {
                near_cell.x = grid_pos.x + x;
                near_cell.y = grid_pos.y + y;
                near_cell.z = grid_pos.z + z;
                value += d_accumulatePointsInCell(grid_point, near_cell, sorted_point, 
                    cell_start, cell_end, limit_dist);
            }
        }
    }

    // 積算された濃度値を格子に格納．
    d_density[address] = value;
}

// 呼び出し関数．
void launchSetDensityToGrid(uint num_voxels, float4 *sorted_point, uint *cell_start, uint *cell_end, float limit_dist, float *d_density)
// uint num_voxels; セルの総数．
// float4 *sorted_point; ソート済み点群．
// unit *cell_start, *cell_end; 各格子セルに属するsorted_pointの部分列の範囲．
// float limit_dist; 打ち切り距離．
// float *d_density; 各格子の濃度値．
{
    int block = 256;
    dim3 grid(num_voxels / block + 1, 1);
    if (grid.x > 65535) {
        grid.y = grid.x / 32768 + 1;
        grid.x = 32768;
    }
    d_setDensityToGrid <<< grid, block >>> (num_voxels, sorted_point, cell_start, cell_end, limit_dist, d_density);
}

// メタボール法により計算された濃度データを保持したCUDAテクスチャ．
texture<float, 1> tex_density;

// 参照テーブルのデータを保持したCUDAテクスチャ．
texture<uint, 1> tex_num_vertices_table;
texture<uint, 1> tex_triangle_table;

// 濃度分布データのCUDAテクスチャへのバインド．
void bindDensityTexture(float *d_density)
// float *d_density; 濃度分布．
{
	cudaChannelFormatDesc desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaBindTexture(0, tex_density, d_density, desc);
}

// セルごとの頂点数の表のCUDAテクスチャへのバインド．
void bindNumVerticesTableTexture(uint *d_num_vertices_table)
// uint *d_num_vertices_table; セルごとの頂点数の表．
{
    cudaChannelFormatDesc desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);
    cudaMemcpy(d_num_vertices_table, numVertsTable, 256 * sizeof(uint), cudaMemcpyHostToDevice);
    cudaBindTexture(0, tex_num_vertices_table, d_num_vertices_table, desc);
}

// 各セルに三角形ポリゴンを貼り付ける際の参照表のCUDAテクスチャへのバインド．
void bindTriangleTableTexture(uint *d_triangle_table)
// uint *d_triangle_table; 各セルに三角形ポリゴンを貼り付ける際の参照表．
{
    cudaChannelFormatDesc desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);
	cudaMemcpy(d_triangle_table, triTable, 256 * 16 * sizeof(uint), cudaMemcpyHostToDevice);
    cudaBindTexture(0, tex_triangle_table, d_triangle_table, desc);
}

// CUDAテクスチャのアンバインド．
void unbindTextures(void)
{
	cudaUnbindTexture(tex_density);
	cudaUnbindTexture(tex_num_vertices_table);
    cudaUnbindTexture(tex_triangle_table);
}

// 与えられた格子点における濃度値．
__device__ float d_sampleDensity(int3 grid_pos)
// int3 grid_pos; 3次元格子におけるグリッド位置．
{
	uint address;
	address = __umul24(__umul24(d_params.grid_size.x, d_params.grid_size.y), grid_pos.z) 
		+ __umul24(d_params.grid_size.x, grid_pos.y) + grid_pos.x;
    return tex1Dfetch(tex_density, address);
}

// 排他的スキャン．
void thrustScan(unsigned int *input, unsigned int num_elements, unsigned int *output)
{
    thrust::exclusive_scan(thrust::device_ptr<unsigned int>(input), 
		thrust::device_ptr<unsigned int>(input + num_elements),
		thrust::device_ptr<unsigned int>(output));
}

// ポリゴンを貼り付けるセルの分類．
__global__ void d_classifyVoxel(uint num_voxels, float iso_value, uint *voxel_vertices, uint *voxel_occupied)
// uint num_voxels; セルの総数．
// float iso_value; 等値面を定める濃度値．
// uint *voxel_veritices; 各セルに貼り付けられる三角形ポリゴンの頂点数．
// uint *vocel_occupied; ポリゴンが貼り付けられるセルのとき1，そうでなければ0．
{
	uint block_index = __umul24(gridDim.x, blockIdx.y) + blockIdx.x; 
	uint index = __umul24(blockDim.x, block_index) + threadIdx.x;
	uint num_vertices, cube_index;
	int3 grid_pos;

    // このスレッドのインデックスはセルの個数をオーバーしている．
    if (index >= num_voxels)
        return;
    grid_pos = d_addressToGridPos(index);

	// 直交格子の端のセルについては，濃度がセットされていない格子があるので無視．
	if ((grid_pos.x >= d_params.grid_size.x - 1) 
		|| (grid_pos.y >= d_params.grid_size.y - 1) 
		|| (grid_pos.z >= d_params.grid_size.z - 1)) {
		voxel_vertices[index] = 0;
		voxel_occupied[index] = false;
	} else {

		// indexが対応するセルの8頂点の濃度値を取得し，その値とiso_valueを
		// 比較．比較結果を8ビットのパターンに登録．
		cube_index =  uint(d_sampleDensity(grid_pos) < iso_value); 
		cube_index += uint(d_sampleDensity(grid_pos + make_int3(1, 0, 0)) < iso_value) * 2; 
		cube_index += uint(d_sampleDensity(grid_pos + make_int3(1, 1, 0)) < iso_value) * 4; 
		cube_index += uint(d_sampleDensity(grid_pos + make_int3(0, 1, 0)) < iso_value) * 8;
		cube_index += uint(d_sampleDensity(grid_pos + make_int3(0, 0, 1)) < iso_value) * 16; 
		cube_index += uint(d_sampleDensity(grid_pos + make_int3(1, 0, 1)) < iso_value) * 32; 
		cube_index += uint(d_sampleDensity(grid_pos + make_int3(1, 1, 1)) < iso_value) * 64; 
		cube_index += uint(d_sampleDensity(grid_pos + make_int3(0, 1, 1)) < iso_value) * 128;

		// 与えられたパターンで頂点数の表を検索し，必要な頂点数を見積もる．
		num_vertices = tex1Dfetch(tex_num_vertices_table, cube_index);
		voxel_vertices[index] = num_vertices;
		voxel_occupied[index] = (num_vertices > 0);
	}
}

// 呼び出し関数．
void launchClassifyVoxel(uint num_voxels, float iso_value, uint *voxel_vertices, uint *voxel_occupied) 
// uint num_voxels; セルの総数．
// float iso_value; 等値面を定める濃度値．
// uint *voxel_veritices; 各セルに貼り付けられる三角形ポリゴンの頂点数．
// unit *vocel_occupied; ポリゴンが貼り付けられるセルのとき1，そうでなければ0．
{
    uint block = 256;
    dim3 grid(num_voxels / block + 1, 1);
    if (grid.x > 65535) {
        grid.y = grid.x / 32768 + 1;
        grid.x = 32768;
    }
    d_classifyVoxel <<< grid, block >>> (num_voxels, iso_value, voxel_vertices, voxel_occupied);
}

// 空セルの圧縮．
__global__ void d_compactVoxels(uint num_voxels, uint *voxel_occupied, uint *voxel_occupied_scanned, 
	uint *compacted_voxel)
// uint num_voxels; セルの総数．
// unit *vocel_occupied; ポリゴンが貼り付けられるセルのとき1，そうでなければ0．
// unit *vocel_occupied_scanned; voxel_occupiedを排他的にスキャンした結果．
// uint *compacted_voxel; 圧縮されたセル．前から順に空ではないセルのインデックス．
{
	uint block_index = __umul24(gridDim.x, blockIdx.y) + blockIdx.x; 
	uint index = __umul24(blockDim.x, block_index) + threadIdx.x;

    // このスレッドのインデックスはセルの個数をオーバーしている．
    if (index >= num_voxels)
        return;
    if (voxel_occupied[index])
        compacted_voxel[voxel_occupied_scanned[index]] = index;
}

// 呼び出し関数．
void launchCompactVoxels(uint num_voxels, uint *voxel_occupied, uint *voxel_occupied_scanned, 
	uint *compacted_voxel)
// uint num_voxels; セルの総数．
// unit *vocel_occupied; ポリゴンが貼り付けられるセルのとき1，そうでなければ0．
// unit *vocel_occupied_scanned; voxel_occupiedを排他的にスキャンした結果．
// uint *compacted_voxel; 圧縮されたセル．前から順に空ではないセルのインデックス．
{    
    uint block = 256;
    dim3 grid(num_voxels / block + 1, 1);
    if (grid.x > 65535) {
        grid.y = grid.x / 32768 + 1;
        grid.x = 32768;
    }
    d_compactVoxels <<< grid, block >>> (num_voxels, voxel_occupied, voxel_occupied_scanned, compacted_voxel);
}

// 線分の内分点の計算．
__device__ float4 d_interpPoint(float iso_value, float4 p0, float4 p1, float f0, float f1)
// float iso_value; 等値面の値．
// float4 p0, p1; 内分する線分の両端点．
// float f0, f1; p0, p1における濃度値．
{
    float t = (iso_value - f0) / (f1 - f0);
    float4 p = (1.0f - t) * p0 + t * p1;
    return p;
} 

// ベクトルの外積．結果は方向ベクトルなのでw要素は常に0.0．
inline __device__ float4 d_cross(float4 a, float4 b)
// float4 a, b; 2つのベクトル．
{
	return make_float4(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x, 0.0f);
}

// ポリゴンの法線の計算．
__device__ float4 d_normal(float4 p0, float4 p1, float4 p2)
// float4 p0, p1, p2; 反時計周りに並ぶ3点の座標．
{
    return(d_cross((p2 - p1), (p0 - p1)));
}

// ポリゴン生成の際に参照する1ブロックあたりのスレッド数．
// シェアードメモリのサイズの制限からあまり大きくできない．
#define NUM_THREADS 64

// セルへの三角形ポリゴンの貼り付け．
__global__ void d_generatePolys(uint num_occupied_voxels, uint max_num_vertices, float iso_value, 
	uint *compacted_voxel, uint *num_vertices_scanned, float4 *poly_pos, float4 *poly_normal)
// uint num_occupied_voxels; ポリゴンが貼り付けられるセルの総数．
// uint max_num_vertices; ポリゴンを構成する頂点数の最大値．
// float iso_value; 等値面の値．
// uint *complacted_voxel; 圧縮されたセル．前から順に空ではないセルのインデックス．
// uint *num_vertices_scanned; セルごとの頂点数の積算．
// float4 *poly_pos, *poly_normal; 出力ポリゴンの座標と法線データ．
{
	uint block_index = __umul24(gridDim.x, blockIdx.y) + blockIdx.x; 
	uint index = __umul24(blockDim.x, block_index) + threadIdx.x;
    uint voxel, cube_index, num_vertices, ed[3];
	uint num_tri_vertices;
	int3 grid_pos;
	float field[8];
	float4 point, v[8], pos[3], normal;
	__shared__ float4 vertex[NUM_THREADS][12];

    // このスレッドのインデックスはセルの個数をオーバーしている．
    if (index >= num_occupied_voxels)
        return;

    // セルの隅の座標値の取得．
    voxel = compacted_voxel[index];
    grid_pos = d_addressToGridPos(voxel);
    point = d_gridPosToPoint(grid_pos);

    // セルの8頂点の座標値の取得．
    v[0] = point;
    v[1] = point + make_float4(d_params.voxel_size.x, 0.0f, 0.0f, 0.0f);
    v[2] = point + make_float4(d_params.voxel_size.x, d_params.voxel_size.y, 0.0f, 0.0f);
    v[3] = point + make_float4(0.0f, d_params.voxel_size.y, 0.0f, 0.0f);
    v[4] = point + make_float4(0.0f, 0.0f, d_params.voxel_size.z, 0.0f);
    v[5] = point + make_float4(d_params.voxel_size.x, 0.0f, d_params.voxel_size.z, 0.0f);
    v[6] = point + make_float4(d_params.voxel_size.x, d_params.voxel_size.y, d_params.voxel_size.z, 0.0f);
    v[7] = point + make_float4(0.0f, d_params.voxel_size.y, d_params.voxel_size.z, 0.0f);

    // セルの8頂点における濃度値の取得．
    field[0] = d_sampleDensity(grid_pos); 
    field[1] = d_sampleDensity(grid_pos + make_int3(1, 0, 0));
    field[2] = d_sampleDensity(grid_pos + make_int3(1, 1, 0));
    field[3] = d_sampleDensity(grid_pos + make_int3(0, 1, 0));
    field[4] = d_sampleDensity(grid_pos + make_int3(0, 0, 1));
    field[5] = d_sampleDensity(grid_pos + make_int3(1, 0, 1));
    field[6] = d_sampleDensity(grid_pos + make_int3(1, 1, 1));
    field[7] = d_sampleDensity(grid_pos + make_int3(0, 1, 1));

    // 8頂点の濃度値とiso_valueの比較．比較結果を8ビットのパターンに登録．
    cube_index =  uint(field[0] < iso_value); 
    cube_index += uint(field[1] < iso_value) * 2; 
    cube_index += uint(field[2] < iso_value) * 4; 
    cube_index += uint(field[3] < iso_value) * 8; 
    cube_index += uint(field[4] < iso_value) * 16; 
    cube_index += uint(field[5] < iso_value) * 32; 
    cube_index += uint(field[6] < iso_value) * 64; 
    cube_index += uint(field[7] < iso_value) * 128;

    // 各辺の両端の濃度値に基づいて12本の辺上の点を取得．ポリゴンの頂点となる．
	vertex[threadIdx.x][0] = d_interpPoint(iso_value, v[0], v[1], field[0], field[1]);
    vertex[threadIdx.x][1] = d_interpPoint(iso_value, v[1], v[2], field[1], field[2]);
    vertex[threadIdx.x][2] = d_interpPoint(iso_value, v[2], v[3], field[2], field[3]);
    vertex[threadIdx.x][3] = d_interpPoint(iso_value, v[3], v[0], field[3], field[0]);
	vertex[threadIdx.x][4] = d_interpPoint(iso_value, v[4], v[5], field[4], field[5]);
    vertex[threadIdx.x][5] = d_interpPoint(iso_value, v[5], v[6], field[5], field[6]);
    vertex[threadIdx.x][6] = d_interpPoint(iso_value, v[6], v[7], field[6], field[7]);
    vertex[threadIdx.x][7] = d_interpPoint(iso_value, v[7], v[4], field[7], field[4]);
	vertex[threadIdx.x][8] = d_interpPoint(iso_value, v[0], v[4], field[0], field[4]);
    vertex[threadIdx.x][9] = d_interpPoint(iso_value, v[1], v[5], field[1], field[5]);
    vertex[threadIdx.x][10] = d_interpPoint(iso_value, v[2], v[6], field[2], field[6]);
    vertex[threadIdx.x][11] = d_interpPoint(iso_value, v[3], v[7], field[3], field[7]);

    // ポリゴンデータの出力．
    num_vertices = tex1Dfetch(tex_num_vertices_table, cube_index);
    for (uint i = 0; i < num_vertices; i += 3) {
        ed[0] = tex1Dfetch(tex_triangle_table, (cube_index * 16) + i);
        pos[0] = vertex[threadIdx.x][ed[0]];
        ed[1] = tex1Dfetch(tex_triangle_table, (cube_index * 16) + i + 1);
        pos[1] = vertex[threadIdx.x][ed[1]];
        ed[2] = tex1Dfetch(tex_triangle_table, (cube_index * 16) + i + 2);
        pos[2] = vertex[threadIdx.x][ed[2]];
		
        // ポリゴンと法線の定義．
		num_tri_vertices = num_vertices_scanned[voxel] + i;
        if (num_tri_vertices < (max_num_vertices - 3)) {
            normal = d_normal(pos[0], pos[1], pos[2]);
			poly_pos[num_tri_vertices] = pos[0]; 
            poly_normal[num_tri_vertices] = normal;
            poly_pos[num_tri_vertices + 1] = pos[1];
            poly_normal[num_tri_vertices + 1] = normal;
            poly_pos[num_tri_vertices + 2] = pos[2];
            poly_normal[num_tri_vertices + 2] = normal; 
        }
    }
}

// 呼び出し関数．
void launchGeneratePolys(uint num_occupied_voxels, uint max_num_vertices, float iso_value, 
	uint *compacted_voxel, uint *num_vertices_scanned, 
	float4 *poly_pos, float4 *poly_norm)
// uint num_occupied_voxels; ポリゴンが貼り付けられるセルの総数．
// uint max_num_vertices; ポリゴンを構成する頂点数の最大値．
// float iso_value; 等値面の値．
// uint *complacted_voxel; 圧縮されたセル．前から順に空ではないセルのインデックス．
// uint *num_vertices_scanned; 圧縮されたセルごとの頂点数．
// float4 *poly_pos, *poly_normal; 出力ポリゴンの座標と法線データ．
{
	dim3 grid(num_occupied_voxels / NUM_THREADS + 1, 1);
    while (grid.x > 65535) {
        grid.x /= 2;
        grid.y *= 2;
    }
    d_generatePolys <<< grid, NUM_THREADS >>> (num_occupied_voxels, max_num_vertices, iso_value,
        compacted_voxel, num_vertices_scanned, poly_pos, poly_norm);
}
