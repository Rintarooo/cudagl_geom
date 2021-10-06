#include <math.h>
#include <stdio.h>

#include <cuda_runtime.h>

// #include <gl/glew.h>
#include <GL/glew.h>
#include <GL/glut.h>

// #include <gl/freeglut.h>

#include "Metaball.h"
#include "BasicDef.h"

// ���f�����ώ@���鎋�_�ʒu�D
#define PHI 30.0
#define THETA 30.0
double phi = PHI;
double theta = THETA;

// �E�B���h�E�̏����ʒu�Ə����T�C�Y�D
#define INIT_X_POS 128
#define INIT_Y_POS 128
#define INIT_WIDTH 512
#define INIT_HEIGHT 512

// �E�B���h�E�̕��ƍ����D
unsigned int window_width = INIT_WIDTH; 
unsigned int window_height = INIT_HEIGHT;

// �`��͈͂̌���̍ۂɈꎞ�I�ɗp����f�[�^�D
double point[MAX_NUM_POINTS][3];
unsigned int num_points;

// �ȉ��̃f�[�^�̓_�~�[�D
unsigned int triangle[MAX_NUM_TRIANGLES][3];
unsigned int num_triangles;

// ���_�ʒu�D
extern double eye[3];

// ���q�`�惂�[�h�D
bool particle_mode = false;

// �ꎞ��~���[�h�D
bool pause_mode = false;

// �}�E�X�����D
int mouse_old_x, mouse_old_y;
bool motion_p;

// ���q�ړ��̎��ԍ��݁D
float anim_dt = 0.1f;

// �ł��؂苗���D
float limit_dist = 0.06f;

// �����Ώۂ̋�Ԃ̒�`�D
float4 world_size = {2.0f, 2.0f, 2.0f, 1.0f};
float4 world_origin = {- 1.0f, - 1.0f, - 1.0f, 1.0f};

// 3�����i�q�̉𑜓x�D
#define GRID_SIZE 128
uint3 grid_size = {GRID_SIZE, GRID_SIZE, GRID_SIZE};
uint num_voxels;
float4 voxel_size;

// ���q�̈ʒu�f�[�^�D
uint num_particles = 100000;
float4 h_position[MAX_NUM_POINTS];
float4 *d_position;

// ���^�{�[���@�ňꎞ�I�ɎQ�Ƃ�����ϐ��D
struct sim_params h_params;
float4 *d_sorted_point;
uint *d_grid_point_address;
uint *d_grid_point_index;
uint *d_voxel_start;
uint *d_voxel_end;
float *d_density;

// �}�[�`���O�L���[�u�@�ňꎞ�I�ɎQ�Ƃ�����ϐ��D
extern uint *d_num_vertices_table;
extern uint *d_triangle_table;
extern uint *d_voxel_vertices;
extern uint *d_voxel_vertices_scanned;
extern uint *d_voxel_occupied;
extern uint *d_voxel_occupied_scanned;
extern uint *d_compacted_voxel;

extern GLuint poly_pos_vbo, poly_normal_vbo;
extern struct cudaGraphicsResource *poly_pos_vbo_res, *poly_normal_vbo_res;
extern uint max_num_vertices;

// �O����`�̊֐��D
extern void setParameters(struct sim_params *h_params);
extern void thrustSortByKey(unsigned int *key, unsigned int *values, unsigned int num_elements);
extern void launchCalcAddress(uint num_points, float4 *point, uint *grid_point_address, uint *grid_point_index);
extern void launchReorderPointsAndAssignPointsToCells(uint num_points, uint num_voxels, float4 *point, 
    uint *grid_point_address, uint *grid_point_index,
    float4 *sorted_point, uint *cell_start, uint *cell_end);
extern void launchSetDensityToGrid(uint num_voxels, float4 *sorted_point, uint *cell_start, uint *cell_end, float limit_dist, float *d_volume);
extern void defineViewMatrix(double phi, double theta, unsigned int width, unsigned int height, double margin);

extern void createVBO(GLuint *vbo, unsigned int size, struct cudaGraphicsResource **vbo_res, unsigned int vbo_res_flags);
extern void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res);
extern void bindDensityTexture(float *d_density);
extern void bindNumVerticesTableTexture(uint *d_num_vertices_table);
extern void bindTriangleTableTexture(uint *d_triangle_table);
extern void unbindTextures(void);
extern void marchingCubesIsoSurface(void);
extern void drawIsoSurface(void);

// ���q�Q�̏����ʒu�ւ̔z�u�D
void setInitialPosition(void)
{
	unsigned int i, j, k;
	float x, y, z;
	srand(12131);
	for (i = 0; i < num_particles; i++) {
		do {
			x = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
			y = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
			z = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
		} while (((x * x + y * y + z * z) > 0.81f) || (fabs(x) > 0.05f));
		h_position[i].x = x;
		h_position[i].y = y;
		h_position[i].z = z;
		h_position[i].w = 1.0f;
	}

	// num_points��point[][3]�̐ݒ�D
	num_points = 8;
	for (i = 0; i < 2; i++) {
		for (j = 0; j < 2; j++) {
			for (k = 0; k < 2; k++) {
				point[4 * i + 2 * j + k][X] = world_origin.x + world_size.x * i;
				point[4 * i + 2 * j + k][Y] = world_origin.y + world_size.y * j;
				point[4 * i + 2 * j + k][Z] = world_origin.z + world_size.z * k;
			}
		}
	}
}

// ���q�̈ړ��D
void moveParticles(void)
{
	unsigned int i;
	double a, x, y;
	for (i = 0; i < num_particles; i++) {
		a = floor(64.0 * h_position[i].z) * anim_dt * (2.0 * PI) * 30.0 / 360.0;
		x = cos(a) * h_position[i].x - sin(a) * h_position[i].y;
		y = sin(a) * h_position[i].x + cos(a) * h_position[i].y;
		h_position[i].x = (float)x;
		h_position[i].y = (float)y;
	}
}

// ���^�{�[���@�ɂ��Z�x��̍X�V�D
void metaballUpdate(void)
{
	
	// ���q�ʒu�̃f�[�^�̃f�o�C�X�������ւ̓]���D
	cudaMemcpy(d_position, h_position, num_particles * sizeof(float4), cudaMemcpyHostToDevice);

    // �e���q�ւ̏�������Z���A�h���X�̊��蓖�āD
    launchCalcAddress(num_particles, d_position, d_grid_point_address, d_grid_point_index);

    // �Z���A�h���X�Ɋ�Â��\�[�g�D
	thrustSortByKey(d_grid_point_address, d_grid_point_index, num_particles);

	// �e�Z���ɑ����闱�q�̌���D
	launchReorderPointsAndAssignPointsToCells(num_particles, num_voxels, d_position, 
        d_grid_point_address, d_grid_point_index, d_sorted_point, d_voxel_start, d_voxel_end);

    // �e�i�q�_�ւ̔Z�x�l�̊��蓖�āD
	launchSetDensityToGrid(num_voxels, d_sorted_point, d_voxel_start, d_voxel_end, limit_dist, d_density);
}

// ���q�Q�̕`��
void drawParticles(void)
{
    uint i;
    glColor3f(0.0f, 1.0f, 0.0f);
    glPointSize(2.0f);
    glBegin(GL_POINTS);
    for (i = 0; i < num_particles; i++)
        glVertex3f(h_position[i].x, h_position[i].y, h_position[i].z);
    glEnd();
}    

// �v�Z���ʂ̕\���D
void display(void)
{
	GLfloat light_pos[4];
 
    // ���^�{�[���@�ɂ��Z�x��̍X�V�D
	metaballUpdate(); 

	// �����e�̐ݒ�D
	defineViewMatrix(phi, theta, window_width, window_height, 0.0);

	// �����̐ݒ�D
	light_pos[0] = (float)eye[X];
	light_pos[1] = (float)eye[Y];
	light_pos[2] = (float)eye[Z];
	light_pos[3] = 0.0f;
	glLightfv(GL_LIGHT0, GL_POSITION, light_pos);

    // ��ʏ����D
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);  

     // �O�g�̕`��D
    glColor3f(1.0f, 1.0f, 1.0f);
    glutWireCube(2.0);

    // ���q�Q�̕`��D
    if (particle_mode)
        drawParticles();

	// �}�[�`���O�L���[�u�@�ɂ��|���S���Q�̕`��D
	else {
		glEnable(GL_LIGHTING);
		glEnable(GL_NORMALIZE);
		marchingCubesIsoSurface();
		drawIsoSurface();
		glDisable(GL_LIGHTING);
	}
	
	// ���q�̈ړ��D
	if (!pause_mode)
		moveParticles();
    glutSwapBuffers();
}

// ���T�C�Y�D
void resize(int width, int height)
{
    window_width = width;
    window_height = height;
}

// CUDA�֌W�̏������D
void initCUDA(void)
{

	// �i�q�\���Ɋւ���p�����[�^�̓]���D
	setParameters(&h_params);

	// ���^�{�[���@�ŗp����f�o�C�X�������̊m�ہD
    cudaMalloc((void**)&d_position, num_particles * sizeof(float4));
    cudaMalloc((void**)&d_sorted_point, num_particles * sizeof(float4));
	cudaMalloc((void**)&d_grid_point_address, num_particles * sizeof(uint));
    cudaMalloc((void**)&d_grid_point_index, num_particles * sizeof(uint));
    cudaMalloc((void**)&d_voxel_start, num_voxels * sizeof(uint));
    cudaMalloc((void**)&d_voxel_end, num_voxels * sizeof(uint));
	cudaMalloc((void**)&d_density, num_voxels * sizeof(float));

    // �}�[�`���O�L���[�u�@�ŗp����f�o�C�X�������̊m�ہD
    cudaMalloc((void**)&d_num_vertices_table, 256 * sizeof(uint));
    cudaMalloc((void**)&d_triangle_table, 256 * 16 * sizeof(uint));
    cudaMalloc((void**)&d_voxel_vertices, num_voxels * sizeof(uint));
    cudaMalloc((void**)&d_voxel_vertices_scanned, num_voxels * sizeof(uint));
    cudaMalloc((void**)&d_voxel_occupied, num_voxels * sizeof(uint));
    cudaMalloc((void**)&d_voxel_occupied_scanned, num_voxels * sizeof(uint));
    cudaMalloc((void**)&d_compacted_voxel, num_voxels * sizeof(uint));

	// �e�N�X�`���������ւ̃o�C���h�D
	bindDensityTexture(d_density);
	bindNumVerticesTableTexture(d_num_vertices_table);
	bindTriangleTableTexture(d_triangle_table);
}

// �㏈���D
void cleanUp(void)
{

    // ���^�{�[���@�ŗp�����f�o�C�X�������̉���D
	cudaFree(d_position);
    cudaFree(d_sorted_point);
    cudaFree(d_grid_point_address);
    cudaFree(d_grid_point_index);
    cudaFree(d_voxel_start);
    cudaFree(d_voxel_end);
	cudaFree(d_density);

    // ���_�o�b�t�@�I�u�W�F�N�g�̏����D
    deleteVBO(&poly_pos_vbo, poly_pos_vbo_res);
	deleteVBO(&poly_normal_vbo, poly_normal_vbo_res);

    // �}�[�`���O�L���[�u�@�ŗp�����f�o�C�X�������̉���D
    cudaFree(d_num_vertices_table);
    cudaFree(d_triangle_table);
    cudaFree(d_voxel_vertices);
    cudaFree(d_voxel_vertices_scanned);
    cudaFree(d_voxel_occupied);
    cudaFree(d_voxel_occupied_scanned);
    cudaFree(d_compacted_voxel);

	// CUDA�e�N�X�`���̃A���o�C���h�D
	unbindTextures();
	cudaDeviceReset();
}

// �L�[�����D
void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
    switch (key) {
        case 'q':
        case 'Q':
		case '\033':
            exit(0);
            break;

			// ���q�\���ƃ|���S���\���̐؂�ւ��D
        case 'p':
        case 'P':
            particle_mode = !particle_mode;
            break;

			// �ꎞ��~�D
        case ' ':
            pause_mode = !pause_mode;
            break;
    }
    glutPostRedisplay();
}

// �}�E�X�̃{�^�������D
void mouse_button(int button, int state, int x, int y)
{
	if ((state == GLUT_DOWN) && (button == GLUT_LEFT_BUTTON))
		motion_p = true;
	else if (state == GLUT_UP)
		motion_p = false;
	mouse_old_x = x;
	mouse_old_y = y;
}

// �}�E�X�̈ړ������D
void mouse_motion(int x, int y)
{
	int dx, dy;
	dx = x - mouse_old_x;
	dy = y - mouse_old_y;
	if (motion_p) {
		phi -= dx * 0.2;
		theta += dy * 0.2;
	}
	mouse_old_x = x;
	mouse_old_y = y;
	glutPostRedisplay();
}

// �A�C�h���D
void idle(void)
{
     glutPostRedisplay();
}

// OpenGL�֌W�̏����ݒ�D
bool initGL(void)
{  
	glClearColor(0.25f, 0.25f, 0.25f, 1.0f);
    glEnable(GL_DEPTH_TEST);
	glClearDepth(1.0);
	glDepthFunc(GL_LESS);
	glEnable(GL_LIGHT0);

	// glew�̏������D     
    glewInit();
    if (!glewIsSupported("GL_VERSION_2_0")) {
		fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        return false;
    }

	// �}�[�`���O�L���[�u�@�ŗp���钸�_�o�b�t�@�I�u�W�F�N�g�̐����D
	createVBO(&poly_pos_vbo, max_num_vertices * 4 * sizeof(float), &poly_pos_vbo_res, cudaGraphicsRegisterFlagsWriteDiscard);
	createVBO(&poly_normal_vbo, max_num_vertices * 4 * sizeof(float), &poly_normal_vbo_res, cudaGraphicsRegisterFlagsWriteDiscard);
	return true;
}

int main(int argc, char** argv) 
{

    // �Z���̊i�q�̉𑜓x�ƃT�C�Y�D
    num_voxels = grid_size.x * grid_size.y * grid_size.z;
    printf("grid: %d x %d x %d = %d voxels\n", grid_size.x, grid_size.y, grid_size.z, num_voxels);
    voxel_size = make_float4(world_size.x / grid_size.x, world_size.y / grid_size.y, world_size.z / grid_size.z, 1.0f);

    // �V�~�����[�V�����p�����[�^�̕ۑ��D
    h_params.grid_size = grid_size;
    h_params.num_voxels = num_voxels;
    h_params.world_origin = world_origin; 
    h_params.voxel_size = voxel_size;

    // �z�肷��ő咸�_���D
    max_num_vertices = grid_size.x * grid_size.y * 100;
    printf("max num vertices = %d\n", max_num_vertices);

	// CUDA�֌W�ł̕K�v�ȃ������̊m�ہD
	initCUDA();

	// ���q�������ʒu�ɐݒ�D
	setInitialPosition();

    // GLUT�̒�`�D
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
	glutInitWindowPosition(INIT_X_POS, INIT_Y_POS);
	glutInitWindowSize(INIT_WIDTH, INIT_HEIGHT);
	glutCreateWindow("Metaball & Marching Cubes");
    glutDisplayFunc(display);
    glutReshapeFunc(resize);
    glutMouseFunc(mouse_button);
    glutMotionFunc(mouse_motion);
    glutKeyboardFunc(keyboard);
    glutIdleFunc(idle);
    atexit(cleanUp);

	// OpenGL�̐ݒ�D
	if (!initGL())
        return 1;
	
    // �A�j���[�V�����`��̃��[�v�D
    glutMainLoop();
    return 0;
}