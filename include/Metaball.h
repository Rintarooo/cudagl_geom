// �^��`�D
typedef unsigned int uint;

// �V�~�����[�V�������ɎQ�Ƃ���p�����[�^�̍\���́D
struct sim_params {
    uint3 grid_size;
    uint num_voxels;
    float4 world_origin;
    float4 voxel_size;
};

// ��̃Z���Ɋi�[�ł��闱�q�̍ő吔�D
#define VOXEL_MARGIN 30


