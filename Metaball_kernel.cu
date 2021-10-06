#include <cuda_runtime.h>

#include <math.h>
// #include "vector_types.h"// cuda vector types, such as float3, float4
#include "cuda_runtime.h"
#include <helper_math.h>


#include "Metaball.h"
#include "tables.h"

// thrust�p�̃w�b�_�[�t�@�C���D
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>

// �i�q�\���̃p�����[�^��ێ������\���́D
__constant__ struct sim_params d_params;

// �p�����[�^�ݒ�D
void setParameters(struct sim_params *h_params)
{
    cudaMemcpyToSymbol(d_params, h_params, sizeof(struct sim_params), 0, cudaMemcpyHostToDevice);
}

// 1�����O���b�h�T�C�Y�̊ȈՌv�Z�D
void computeGridSize(uint n, uint block_size, uint *grid_size, uint *new_block_size)
// unit n; �����Ώۂ̑����D
// unit block_size; 1�u���b�N������̃X���b�h���D
// uint &grid_size; �O���b�h�T�C�Y�D
// uint &new_block_size; �␳���ꂽ1�u���b�N������̃X���b�h���D
{
    *new_block_size = min(block_size, n);
    *grid_size = ((n % *new_block_size == 0) ? (n / *new_block_size) : (n / *new_block_size + 1));
}

// ���W��3�����i�q�ɂ�����O���b�h�ʒu�D
__device__ int3 d_pointToGridPos(float4 point)
// float4 point; ���W�l�D
{
    int3 grid_pos;
    grid_pos.x = (int)floor((point.x - d_params.world_origin.x) / d_params.voxel_size.x);
    grid_pos.y = (int)floor((point.y - d_params.world_origin.y) / d_params.voxel_size.y);
    grid_pos.z = (int)floor((point.z - d_params.world_origin.z) / d_params.voxel_size.z);
    return grid_pos;
}

// �O���b�h�ʒu�ɂ����邨������W�D
__device__ float4 d_gridPosToPoint(int3 grid_pos)
// int3 grid_pos; �O���b�h�ʒu�D
{
    float4 p;
    p.x = d_params.world_origin.x + grid_pos.x * d_params.voxel_size.x;
    p.y = d_params.world_origin.y + grid_pos.y * d_params.voxel_size.y;
    p.z = d_params.world_origin.z + grid_pos.z * d_params.voxel_size.z;
    p.w = 1.0f;
    return p;
}    

// 3�����I�ȃO���b�h�ʒu����1�����I�ȃZ���A�h���X�ւ̕ϊ��D
__device__ uint d_gridPosToAddress(int3 grid_pos)
// int3 grid_pos; �O���b�h�ʒu�D
{

    // �ʒu���i�q�͈͓̔��Ɏ��܂�悤�ɕ␳�D
    if (grid_pos.x < 0) grid_pos.x = 0;
    if (grid_pos.x > (d_params.grid_size.x - 1)) grid_pos.x = (d_params.grid_size.x - 1);
    if (grid_pos.y < 0) grid_pos.y = 0;
    if (grid_pos.y > (d_params.grid_size.y - 1)) grid_pos.y = (d_params.grid_size.y - 1);
    if (grid_pos.z < 0) grid_pos.z = 0;
    if (grid_pos.z > (d_params.grid_size.z - 1)) grid_pos.z = (d_params.grid_size.z - 1);

    // 1�����I�ȃZ���A�h���X�̌v�Z�D
    return __umul24(__umul24(d_params.grid_size.x, d_params.grid_size.y), grid_pos.z) + __umul24(d_params.grid_size.x, grid_pos.y) + grid_pos.x;
}

// 3�����I�ȃO���b�h�ʒu���i�q�͈͓̔����ǂ����̔���D
__device__ bool d_validGridPosP(int3 grid_pos)
// int3 grid_pos; �O���b�h�ʒu�D
{
	return((grid_pos.x >= 0) && (grid_pos.x < d_params.grid_size.x) 
		&& (grid_pos.y >= 0) && (grid_pos.y < d_params.grid_size.y)
		&& (grid_pos.z >= 0) && (grid_pos.z < d_params.grid_size.z));
}

// �Z���A�h���X����O���b�h�ʒu�ւ̕ϊ��D
__device__ int3 d_addressToGridPos(uint address)
// uint address; �Z���A�h���X�D
{
	uint tmp;
    int3 grid_pos;
    grid_pos.z = address / (d_params.grid_size.x * d_params.grid_size.y);
    tmp = address % (d_params.grid_size.x * d_params.grid_size.y);
    grid_pos.y = tmp / d_params.grid_size.x;
    grid_pos.x = tmp % d_params.grid_size.x; 
    return grid_pos;
}

// ���W�ւ̃Z���A�h���X�̊��蓖�āD
__global__ void d_calcAddress(uint num_points, float4 *point, uint *grid_point_address, uint *grid_point_index)
// uint num_points; �_�̑����D
// float4 *point; �_�Q�D
// unit *grid_point_address; �e�_��3�����i�q�ɂ�����Z���A�h���X�D
// unit *grid_point_index; �e�_�̃C���f�b�N�X�D
{
    uint index, address;

    // �����Ώۂ̓_�̑I���D
	index = __umul24(blockDim.x, blockIdx.x) + threadIdx.x;
    if (index >= num_points) 
        return;

    // �Z���A�h���X�̎擾�D
	address = d_gridPosToAddress(d_pointToGridPos(point[index]));

    // ����ꂽ�A�h���X��_�̃C���f�b�N�X�ƂƂ��Ɋi�[�D
    grid_point_address[index] = address;
    grid_point_index[index] = index;
}

// �Ăяo���֐��D
void launchCalcAddress(uint num_points, float4 *point, uint *grid_point_address, uint *grid_point_index)
// uint num_points; �_�̑����D
// float4 *point; �_�Q�D
// unit *grid_point_address; �e�_��3�����i�q�ɂ�����Z���A�h���X�D
// unit *grid_point_index; �e�_�̃C���f�b�N�X�D
{
    uint grid, block;
    computeGridSize(num_points, 256, &grid, &block);
    d_calcAddress <<< grid, block >>> (num_points, point, grid_point_address, grid_point_index);
}

// �L�[�Ɋ�Â��\�[�g�D
void thrustSortByKey(unsigned int *key, unsigned int *values, unsigned int num_elements)
{
    thrust::sort_by_key(thrust::device_ptr<unsigned int>(key), 
		thrust::device_ptr<unsigned int>(key + num_elements),
		thrust::device_ptr<unsigned int>(values));
}

// �e�i�q�Z���ɑ�����_�̕�����̎擾�D
__global__ void d_reorderPointsAndAssignPointsToCells(uint num_points, float4 *point, 
	uint *grid_point_address, uint *grid_point_index,
    float4 *sorted_point, uint *cell_start, uint *cell_end)
// uint num_points; �_�̑����D
// float4 *point; �_�Q�D
// unit *grid_point_address; �e�_�̃Z���A�h���X�D�A�h���X�ɏ]���ă\�[�g�ς݁D
// unit *grid_point_index; �e�_�̃C���f�b�N�X�D�A�h���X�ɏ]���ă\�[�g�ς݁D
// float4 *sorted_point; �\�[�g�ς݂̓_�Q�D
// unit *cell_start, *cell_end; �e�i�q�Z���ɑ�����sorted_point�̕�����͈̔́D
{
	extern __shared__ uint s_address[]; // ���L�������̓u���b�N�T�C�Y+1�̑傫���ɊO����`����Ă���D
    uint index, address;

    // �����Ώۂ̓_�̑I���D
	index = __umul24(blockDim.x, blockIdx.x) + threadIdx.x;
    if (index >= num_points) 
        return;

	// �ׂ̃X���b�h�ł��̓_�̃A�h���X���Q�Ƃł���悤�ɁC����炵�ċ��L
	// �������Ɋi�[�D
	address = grid_point_address[index];
	s_address[threadIdx.x + 1] = address;
	if ((index > 0) && (threadIdx.x == 0)) {
		    
		// ���L�������̐擪�ɂ͈�O�̃u���b�N�̍Ō�̃A�h���X���i�[�D
		s_address[0] = grid_point_address[index - 1];
	}
	__syncthreads();	

	// ��O�̓_�Ƃ��̓_�ƂŃA�h���X�̐؂�ւ�肪�������D
	if ((index == 0) || (address != s_address[threadIdx.x])) {
		cell_start[address] = index;
		if (index > 0)
			 cell_end[s_address[threadIdx.x]] = index;
	}
	if (index == (num_points - 1))
		cell_end[address] = num_points;

	// �_���\�[�g���ɂ��܂������Ă����D
	sorted_point[index] = point[grid_point_index[index]];
}

// �Ăяo���֐��D
void launchReorderPointsAndAssignPointsToCells(uint num_points, uint num_voxels, float4 *point,
	uint *grid_point_address, uint *grid_point_index,
	float4 *sorted_point, uint *cell_start, uint *cell_end)
// uint num_points; �_�̑����D
// uint num_voxels; �Z���̑����D
// float4 *point; �_�Q�D
// unit *grid_point_address; �e�_�̃Z���A�h���X�D�A�h���X�ɏ]���ă\�[�g�ς݁D
// unit *grid_point_index; �e�_�̃C���f�b�N�X�D�A�h���X�ɏ]���ă\�[�g�ς݁D
// float4 *sorted_point; �\�[�g�ς݂̓_�Q�D
// unit *cell_start, *cell_end; �e�i�q�Z���ɑ�����sorted_point�̕�����͈̔́D
{
    uint grid, block;

    // �\�ߑS�Ă�cell_start��0xffffffff�����܂��Ă����D
	cudaMemset(cell_start, 0xffffffff, num_voxels * sizeof(uint));

    // ���L�������T�C�Y�̊O����`�D
	computeGridSize(num_points, 256, &grid, &block);  
	uint s_mem_size = (block + 1) * sizeof(uint);
    d_reorderPointsAndAssignPointsToCells <<< grid, block, s_mem_size >>> (num_points,
        point, grid_point_address, grid_point_index, sorted_point, cell_start, cell_end);
}

// �i�q�_�ɂ��ăZ�����̗��q�̔Z�x��ώZ�D
__device__ float d_accumulatePointsInCell(float4 grid_point, int3 grid_pos, float4* sorted_point, 
	uint *cell_start, uint *cell_end, float limit_dist)
// float4 grid_point; �i�q��̓_�D
// int3 grid_pos; �Z���̃O���b�h�ʒu�D���̃Z�����̓_�̉e����ώZ�D
// float4 *sorted_point; �\�[�g�ς݂̓_�Q�D
// unit *cell_start, *cell_end; �e�i�q�Z���ɑ�����sorted_point�̕�����͈̔́D
// float limit_dist; �ł��؂苗���D
{
	uint i;
    uint address = d_gridPosToAddress(grid_pos);
	float value = 0.0f, d2;
    float4 rel_pos;

    // �Z���ɓ_���܂܂�Ă���D
    if (d_validGridPosP(grid_pos) && (cell_start[address] != 0xffffffff)) {

        // ���̃Z�����̓_�̉e����ώZ�D
		for(i = cell_start[address]; i < cell_end[address]; i++) {
 
            // �_�Ɗi�q��̓_�̑��Έʒu���v�Z�D
            rel_pos = sorted_point[i] - grid_point;
            d2 = rel_pos.x * rel_pos.x + rel_pos.y * rel_pos.y + rel_pos.z * rel_pos.z;
            if (d2 < limit_dist * limit_dist) // �ł��؂苗�����ł���ΐώZ�D
                value += 0.0002f / d2;
        }
    }
    return value;
}

// �e�i�q�_�ւ̔Z�x�l�̊��蓖�āD
__global__ void d_setDensityToGrid(uint num_voxels, float4 *sorted_point, uint *cell_start, uint *cell_end, float limit_dist, float *d_density)
// uint num_voxels; �Z���̑����D
// float4 *sorted_point; �\�[�g�ς݂̓_�Q�D
// unit *cell_start, *cell_end; �e�i�q�Z���ɑ�����sorted_point�̕�����͈̔́D
// float limit_dist; �ł��؂苗���D
// float *d_density; �e�i�q�̔Z�x�l�D
{
    uint block_index = __umul24(gridDim.x, blockIdx.y) + blockIdx.x; 
    uint address = __umul24(blockDim.x, block_index) + threadIdx.x;
	int x, y, z;
    int3 near_cell, grid_range;
	int3 grid_pos;
	float value;
	float4 grid_point;

    // ���̃X���b�h�̃C���f�b�N�X�i���Z���A�h���X�j�̓Z���̑������I�[�o�[���Ă���D
    if (address >= num_voxels) 
        return;

	// �`�F�b�N����Z���͈̔́D1���]�܂����D
	grid_range.x = (int)ceil(limit_dist / d_params.voxel_size.x);
	grid_range.y = (int)ceil(limit_dist / d_params.voxel_size.y);
	grid_range.z = (int)ceil(limit_dist / d_params.voxel_size.z);

    // �i�q�_�̃O���b�h�ʒu�𓾂�D
    grid_pos = d_addressToGridPos(address);

    // �i�q�̍��W�̎擾�D
    grid_point = d_gridPosToPoint(grid_pos);

    // �ߖT�̃Z�����̓_�ɂ��Z�x�l��ώZ�D
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

    // �ώZ���ꂽ�Z�x�l���i�q�Ɋi�[�D
    d_density[address] = value;
}

// �Ăяo���֐��D
void launchSetDensityToGrid(uint num_voxels, float4 *sorted_point, uint *cell_start, uint *cell_end, float limit_dist, float *d_density)
// uint num_voxels; �Z���̑����D
// float4 *sorted_point; �\�[�g�ςݓ_�Q�D
// unit *cell_start, *cell_end; �e�i�q�Z���ɑ�����sorted_point�̕�����͈̔́D
// float limit_dist; �ł��؂苗���D
// float *d_density; �e�i�q�̔Z�x�l�D
{
    int block = 256;
    dim3 grid(num_voxels / block + 1, 1);
    if (grid.x > 65535) {
        grid.y = grid.x / 32768 + 1;
        grid.x = 32768;
    }
    d_setDensityToGrid <<< grid, block >>> (num_voxels, sorted_point, cell_start, cell_end, limit_dist, d_density);
}

// ���^�{�[���@�ɂ��v�Z���ꂽ�Z�x�f�[�^��ێ�����CUDA�e�N�X�`���D
texture<float, 1> tex_density;

// �Q�ƃe�[�u���̃f�[�^��ێ�����CUDA�e�N�X�`���D
texture<uint, 1> tex_num_vertices_table;
texture<uint, 1> tex_triangle_table;

// �Z�x���z�f�[�^��CUDA�e�N�X�`���ւ̃o�C���h�D
void bindDensityTexture(float *d_density)
// float *d_density; �Z�x���z�D
{
	cudaChannelFormatDesc desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaBindTexture(0, tex_density, d_density, desc);
}

// �Z�����Ƃ̒��_���̕\��CUDA�e�N�X�`���ւ̃o�C���h�D
void bindNumVerticesTableTexture(uint *d_num_vertices_table)
// uint *d_num_vertices_table; �Z�����Ƃ̒��_���̕\�D
{
    cudaChannelFormatDesc desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);
    cudaMemcpy(d_num_vertices_table, numVertsTable, 256 * sizeof(uint), cudaMemcpyHostToDevice);
    cudaBindTexture(0, tex_num_vertices_table, d_num_vertices_table, desc);
}

// �e�Z���ɎO�p�`�|���S����\��t����ۂ̎Q�ƕ\��CUDA�e�N�X�`���ւ̃o�C���h�D
void bindTriangleTableTexture(uint *d_triangle_table)
// uint *d_triangle_table; �e�Z���ɎO�p�`�|���S����\��t����ۂ̎Q�ƕ\�D
{
    cudaChannelFormatDesc desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);
	cudaMemcpy(d_triangle_table, triTable, 256 * 16 * sizeof(uint), cudaMemcpyHostToDevice);
    cudaBindTexture(0, tex_triangle_table, d_triangle_table, desc);
}

// CUDA�e�N�X�`���̃A���o�C���h�D
void unbindTextures(void)
{
	cudaUnbindTexture(tex_density);
	cudaUnbindTexture(tex_num_vertices_table);
    cudaUnbindTexture(tex_triangle_table);
}

// �^����ꂽ�i�q�_�ɂ�����Z�x�l�D
__device__ float d_sampleDensity(int3 grid_pos)
// int3 grid_pos; 3�����i�q�ɂ�����O���b�h�ʒu�D
{
	uint address;
	address = __umul24(__umul24(d_params.grid_size.x, d_params.grid_size.y), grid_pos.z) 
		+ __umul24(d_params.grid_size.x, grid_pos.y) + grid_pos.x;
    return tex1Dfetch(tex_density, address);
}

// �r���I�X�L�����D
void thrustScan(unsigned int *input, unsigned int num_elements, unsigned int *output)
{
    thrust::exclusive_scan(thrust::device_ptr<unsigned int>(input), 
		thrust::device_ptr<unsigned int>(input + num_elements),
		thrust::device_ptr<unsigned int>(output));
}

// �|���S����\��t����Z���̕��ށD
__global__ void d_classifyVoxel(uint num_voxels, float iso_value, uint *voxel_vertices, uint *voxel_occupied)
// uint num_voxels; �Z���̑����D
// float iso_value; ���l�ʂ��߂�Z�x�l�D
// uint *voxel_veritices; �e�Z���ɓ\��t������O�p�`�|���S���̒��_���D
// uint *vocel_occupied; �|���S�����\��t������Z���̂Ƃ�1�C�����łȂ����0�D
{
	uint block_index = __umul24(gridDim.x, blockIdx.y) + blockIdx.x; 
	uint index = __umul24(blockDim.x, block_index) + threadIdx.x;
	uint num_vertices, cube_index;
	int3 grid_pos;

    // ���̃X���b�h�̃C���f�b�N�X�̓Z���̌����I�[�o�[���Ă���D
    if (index >= num_voxels)
        return;
    grid_pos = d_addressToGridPos(index);

	// �����i�q�̒[�̃Z���ɂ��ẮC�Z�x���Z�b�g����Ă��Ȃ��i�q������̂Ŗ����D
	if ((grid_pos.x >= d_params.grid_size.x - 1) 
		|| (grid_pos.y >= d_params.grid_size.y - 1) 
		|| (grid_pos.z >= d_params.grid_size.z - 1)) {
		voxel_vertices[index] = 0;
		voxel_occupied[index] = false;
	} else {

		// index���Ή�����Z����8���_�̔Z�x�l���擾���C���̒l��iso_value��
		// ��r�D��r���ʂ�8�r�b�g�̃p�^�[���ɓo�^�D
		cube_index =  uint(d_sampleDensity(grid_pos) < iso_value); 
		cube_index += uint(d_sampleDensity(grid_pos + make_int3(1, 0, 0)) < iso_value) * 2; 
		cube_index += uint(d_sampleDensity(grid_pos + make_int3(1, 1, 0)) < iso_value) * 4; 
		cube_index += uint(d_sampleDensity(grid_pos + make_int3(0, 1, 0)) < iso_value) * 8;
		cube_index += uint(d_sampleDensity(grid_pos + make_int3(0, 0, 1)) < iso_value) * 16; 
		cube_index += uint(d_sampleDensity(grid_pos + make_int3(1, 0, 1)) < iso_value) * 32; 
		cube_index += uint(d_sampleDensity(grid_pos + make_int3(1, 1, 1)) < iso_value) * 64; 
		cube_index += uint(d_sampleDensity(grid_pos + make_int3(0, 1, 1)) < iso_value) * 128;

		// �^����ꂽ�p�^�[���Œ��_���̕\���������C�K�v�Ȓ��_�������ς���D
		num_vertices = tex1Dfetch(tex_num_vertices_table, cube_index);
		voxel_vertices[index] = num_vertices;
		voxel_occupied[index] = (num_vertices > 0);
	}
}

// �Ăяo���֐��D
void launchClassifyVoxel(uint num_voxels, float iso_value, uint *voxel_vertices, uint *voxel_occupied) 
// uint num_voxels; �Z���̑����D
// float iso_value; ���l�ʂ��߂�Z�x�l�D
// uint *voxel_veritices; �e�Z���ɓ\��t������O�p�`�|���S���̒��_���D
// unit *vocel_occupied; �|���S�����\��t������Z���̂Ƃ�1�C�����łȂ����0�D
{
    uint block = 256;
    dim3 grid(num_voxels / block + 1, 1);
    if (grid.x > 65535) {
        grid.y = grid.x / 32768 + 1;
        grid.x = 32768;
    }
    d_classifyVoxel <<< grid, block >>> (num_voxels, iso_value, voxel_vertices, voxel_occupied);
}

// ��Z���̈��k�D
__global__ void d_compactVoxels(uint num_voxels, uint *voxel_occupied, uint *voxel_occupied_scanned, 
	uint *compacted_voxel)
// uint num_voxels; �Z���̑����D
// unit *vocel_occupied; �|���S�����\��t������Z���̂Ƃ�1�C�����łȂ����0�D
// unit *vocel_occupied_scanned; voxel_occupied��r���I�ɃX�L�����������ʁD
// uint *compacted_voxel; ���k���ꂽ�Z���D�O���珇�ɋ�ł͂Ȃ��Z���̃C���f�b�N�X�D
{
	uint block_index = __umul24(gridDim.x, blockIdx.y) + blockIdx.x; 
	uint index = __umul24(blockDim.x, block_index) + threadIdx.x;

    // ���̃X���b�h�̃C���f�b�N�X�̓Z���̌����I�[�o�[���Ă���D
    if (index >= num_voxels)
        return;
    if (voxel_occupied[index])
        compacted_voxel[voxel_occupied_scanned[index]] = index;
}

// �Ăяo���֐��D
void launchCompactVoxels(uint num_voxels, uint *voxel_occupied, uint *voxel_occupied_scanned, 
	uint *compacted_voxel)
// uint num_voxels; �Z���̑����D
// unit *vocel_occupied; �|���S�����\��t������Z���̂Ƃ�1�C�����łȂ����0�D
// unit *vocel_occupied_scanned; voxel_occupied��r���I�ɃX�L�����������ʁD
// uint *compacted_voxel; ���k���ꂽ�Z���D�O���珇�ɋ�ł͂Ȃ��Z���̃C���f�b�N�X�D
{    
    uint block = 256;
    dim3 grid(num_voxels / block + 1, 1);
    if (grid.x > 65535) {
        grid.y = grid.x / 32768 + 1;
        grid.x = 32768;
    }
    d_compactVoxels <<< grid, block >>> (num_voxels, voxel_occupied, voxel_occupied_scanned, compacted_voxel);
}

// �����̓����_�̌v�Z�D
__device__ float4 d_interpPoint(float iso_value, float4 p0, float4 p1, float f0, float f1)
// float iso_value; ���l�ʂ̒l�D
// float4 p0, p1; ������������̗��[�_�D
// float f0, f1; p0, p1�ɂ�����Z�x�l�D
{
    float t = (iso_value - f0) / (f1 - f0);
    float4 p = (1.0f - t) * p0 + t * p1;
    return p;
} 

// �x�N�g���̊O�ρD���ʂ͕����x�N�g���Ȃ̂�w�v�f�͏��0.0�D
inline __device__ float4 d_cross(float4 a, float4 b)
// float4 a, b; 2�̃x�N�g���D
{
	return make_float4(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x, 0.0f);
}

// �|���S���̖@���̌v�Z�D
__device__ float4 d_normal(float4 p0, float4 p1, float4 p2)
// float4 p0, p1, p2; �����v����ɕ���3�_�̍��W�D
{
    return(d_cross((p2 - p1), (p0 - p1)));
}

// �|���S�������̍ۂɎQ�Ƃ���1�u���b�N������̃X���b�h���D
// �V�F�A�[�h�������̃T�C�Y�̐������炠�܂�傫���ł��Ȃ��D
#define NUM_THREADS 64

// �Z���ւ̎O�p�`�|���S���̓\��t���D
__global__ void d_generatePolys(uint num_occupied_voxels, uint max_num_vertices, float iso_value, 
	uint *compacted_voxel, uint *num_vertices_scanned, float4 *poly_pos, float4 *poly_normal)
// uint num_occupied_voxels; �|���S�����\��t������Z���̑����D
// uint max_num_vertices; �|���S�����\�����钸�_���̍ő�l�D
// float iso_value; ���l�ʂ̒l�D
// uint *complacted_voxel; ���k���ꂽ�Z���D�O���珇�ɋ�ł͂Ȃ��Z���̃C���f�b�N�X�D
// uint *num_vertices_scanned; �Z�����Ƃ̒��_���̐ώZ�D
// float4 *poly_pos, *poly_normal; �o�̓|���S���̍��W�Ɩ@���f�[�^�D
{
	uint block_index = __umul24(gridDim.x, blockIdx.y) + blockIdx.x; 
	uint index = __umul24(blockDim.x, block_index) + threadIdx.x;
    uint voxel, cube_index, num_vertices, ed[3];
	uint num_tri_vertices;
	int3 grid_pos;
	float field[8];
	float4 point, v[8], pos[3], normal;
	__shared__ float4 vertex[NUM_THREADS][12];

    // ���̃X���b�h�̃C���f�b�N�X�̓Z���̌����I�[�o�[���Ă���D
    if (index >= num_occupied_voxels)
        return;

    // �Z���̋��̍��W�l�̎擾�D
    voxel = compacted_voxel[index];
    grid_pos = d_addressToGridPos(voxel);
    point = d_gridPosToPoint(grid_pos);

    // �Z����8���_�̍��W�l�̎擾�D
    v[0] = point;
    v[1] = point + make_float4(d_params.voxel_size.x, 0.0f, 0.0f, 0.0f);
    v[2] = point + make_float4(d_params.voxel_size.x, d_params.voxel_size.y, 0.0f, 0.0f);
    v[3] = point + make_float4(0.0f, d_params.voxel_size.y, 0.0f, 0.0f);
    v[4] = point + make_float4(0.0f, 0.0f, d_params.voxel_size.z, 0.0f);
    v[5] = point + make_float4(d_params.voxel_size.x, 0.0f, d_params.voxel_size.z, 0.0f);
    v[6] = point + make_float4(d_params.voxel_size.x, d_params.voxel_size.y, d_params.voxel_size.z, 0.0f);
    v[7] = point + make_float4(0.0f, d_params.voxel_size.y, d_params.voxel_size.z, 0.0f);

    // �Z����8���_�ɂ�����Z�x�l�̎擾�D
    field[0] = d_sampleDensity(grid_pos); 
    field[1] = d_sampleDensity(grid_pos + make_int3(1, 0, 0));
    field[2] = d_sampleDensity(grid_pos + make_int3(1, 1, 0));
    field[3] = d_sampleDensity(grid_pos + make_int3(0, 1, 0));
    field[4] = d_sampleDensity(grid_pos + make_int3(0, 0, 1));
    field[5] = d_sampleDensity(grid_pos + make_int3(1, 0, 1));
    field[6] = d_sampleDensity(grid_pos + make_int3(1, 1, 1));
    field[7] = d_sampleDensity(grid_pos + make_int3(0, 1, 1));

    // 8���_�̔Z�x�l��iso_value�̔�r�D��r���ʂ�8�r�b�g�̃p�^�[���ɓo�^�D
    cube_index =  uint(field[0] < iso_value); 
    cube_index += uint(field[1] < iso_value) * 2; 
    cube_index += uint(field[2] < iso_value) * 4; 
    cube_index += uint(field[3] < iso_value) * 8; 
    cube_index += uint(field[4] < iso_value) * 16; 
    cube_index += uint(field[5] < iso_value) * 32; 
    cube_index += uint(field[6] < iso_value) * 64; 
    cube_index += uint(field[7] < iso_value) * 128;

    // �e�ӂ̗��[�̔Z�x�l�Ɋ�Â���12�{�̕ӏ�̓_���擾�D�|���S���̒��_�ƂȂ�D
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

    // �|���S���f�[�^�̏o�́D
    num_vertices = tex1Dfetch(tex_num_vertices_table, cube_index);
    for (uint i = 0; i < num_vertices; i += 3) {
        ed[0] = tex1Dfetch(tex_triangle_table, (cube_index * 16) + i);
        pos[0] = vertex[threadIdx.x][ed[0]];
        ed[1] = tex1Dfetch(tex_triangle_table, (cube_index * 16) + i + 1);
        pos[1] = vertex[threadIdx.x][ed[1]];
        ed[2] = tex1Dfetch(tex_triangle_table, (cube_index * 16) + i + 2);
        pos[2] = vertex[threadIdx.x][ed[2]];
		
        // �|���S���Ɩ@���̒�`�D
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

// �Ăяo���֐��D
void launchGeneratePolys(uint num_occupied_voxels, uint max_num_vertices, float iso_value, 
	uint *compacted_voxel, uint *num_vertices_scanned, 
	float4 *poly_pos, float4 *poly_norm)
// uint num_occupied_voxels; �|���S�����\��t������Z���̑����D
// uint max_num_vertices; �|���S�����\�����钸�_���̍ő�l�D
// float iso_value; ���l�ʂ̒l�D
// uint *complacted_voxel; ���k���ꂽ�Z���D�O���珇�ɋ�ł͂Ȃ��Z���̃C���f�b�N�X�D
// uint *num_vertices_scanned; ���k���ꂽ�Z�����Ƃ̒��_���D
// float4 *poly_pos, *poly_normal; �o�̓|���S���̍��W�Ɩ@���f�[�^�D
{
	dim3 grid(num_occupied_voxels / NUM_THREADS + 1, 1);
    while (grid.x > 65535) {
        grid.x /= 2;
        grid.y *= 2;
    }
    d_generatePolys <<< grid, NUM_THREADS >>> (num_occupied_voxels, max_num_vertices, iso_value,
        compacted_voxel, num_vertices_scanned, poly_pos, poly_norm);
}
