// ���W�l��W�����Q�Ƃ���ۂ̃C���f�b�N�X�D
#define X 0
#define Y 1
#define Z 2

#define A 0
#define B 1
#define C 2
#define D 3

// �\���ɔ����Ȓl�D
#define EPS 0.00001

// �\���ɑ傫�Ȓl�D
#define LARGE 100000.0

// �~�����D
#define PI 3.141592653589793

// �}�`�v�f�̍ő���D
#define MAX_NUM_POINTS 2000000
#define MAX_NUM_EDGES 5000000
#define MAX_NUM_TRIANGLES 2000000

// �ꎞ�I�ɗ��p����_�̍\���́D
typedef struct tmp_point {
	double coord[3];	// ���W�D
	unsigned int index; // �_�ɕt�����郆�j�[�N�ȃC���f�b�N�X�D
} tmp_point;

// �ꎞ�I�ɗ��p����ӂ̍\���́D
typedef struct tmp_edge {
	unsigned int start;
	unsigned int end;
	tmp_edge *next;
} tmp_edge;