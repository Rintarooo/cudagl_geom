// 型定義．
typedef unsigned int uint;

// シミュレーション時に参照するパラメータの構造体．
struct sim_params {
    uint3 grid_size;
    uint num_voxels;
    float4 world_origin;
    float4 voxel_size;
};

// 一つのセルに格納できる粒子の最大数．
#define VOXEL_MARGIN 30


