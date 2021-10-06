#include <math.h>
#include <stdio.h>

#include <cuda_runtime.h>

// #include <gl/glew.h>
#include <GL/glew.h>
#include <GL/glut.h>

// #include <gl/freeglut.h>

#include "Metaball.h"
#include "BasicDef.h"

// モデルを観察する視点位置．
#define PHI 30.0
#define THETA 30.0
double phi = PHI;
double theta = THETA;

// ウィンドウの初期位置と初期サイズ．
#define INIT_X_POS 128
#define INIT_Y_POS 128
#define INIT_WIDTH 512
#define INIT_HEIGHT 512

// ウィンドウの幅と高さ．
unsigned int window_width = INIT_WIDTH; 
unsigned int window_height = INIT_HEIGHT;

// 描画範囲の決定の際に一時的に用いるデータ．
double point[MAX_NUM_POINTS][3];
unsigned int num_points;

// 以下のデータはダミー．
unsigned int triangle[MAX_NUM_TRIANGLES][3];
unsigned int num_triangles;

// 視点位置．
extern double eye[3];

// 粒子描画モード．
bool particle_mode = false;

// 一時停止モード．
bool pause_mode = false;

// マウス処理．
int mouse_old_x, mouse_old_y;
bool motion_p;

// 粒子移動の時間刻み．
float anim_dt = 0.1f;

// 打ち切り距離．
float limit_dist = 0.06f;

// 処理対象の空間の定義．
float4 world_size = {2.0f, 2.0f, 2.0f, 1.0f};
float4 world_origin = {- 1.0f, - 1.0f, - 1.0f, 1.0f};

// 3次元格子の解像度．
#define GRID_SIZE 128
uint3 grid_size = {GRID_SIZE, GRID_SIZE, GRID_SIZE};
uint num_voxels;
float4 voxel_size;

// 粒子の位置データ．
uint num_particles = 100000;
float4 h_position[MAX_NUM_POINTS];
float4 *d_position;

// メタボール法で一時的に参照する大域変数．
struct sim_params h_params;
float4 *d_sorted_point;
uint *d_grid_point_address;
uint *d_grid_point_index;
uint *d_voxel_start;
uint *d_voxel_end;
float *d_density;

// マーチングキューブ法で一時的に参照する大域変数．
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

// 外部定義の関数．
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

// 粒子群の初期位置への配置．
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

	// num_pointsとpoint[][3]の設定．
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

// 粒子の移動．
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

// メタボール法による濃度場の更新．
void metaballUpdate(void)
{
	
	// 粒子位置のデータのデバイスメモリへの転送．
	cudaMemcpy(d_position, h_position, num_particles * sizeof(float4), cudaMemcpyHostToDevice);

    // 各粒子への所属するセルアドレスの割り当て．
    launchCalcAddress(num_particles, d_position, d_grid_point_address, d_grid_point_index);

    // セルアドレスに基づくソート．
	thrustSortByKey(d_grid_point_address, d_grid_point_index, num_particles);

	// 各セルに属する粒子の決定．
	launchReorderPointsAndAssignPointsToCells(num_particles, num_voxels, d_position, 
        d_grid_point_address, d_grid_point_index, d_sorted_point, d_voxel_start, d_voxel_end);

    // 各格子点への濃度値の割り当て．
	launchSetDensityToGrid(num_voxels, d_sorted_point, d_voxel_start, d_voxel_end, limit_dist, d_density);
}

// 粒子群の描画
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

// 計算結果の表示．
void display(void)
{
	GLfloat light_pos[4];
 
    // メタボール法による濃度場の更新．
	metaballUpdate(); 

	// 正投影の設定．
	defineViewMatrix(phi, theta, window_width, window_height, 0.0);

	// 光源の設定．
	light_pos[0] = (float)eye[X];
	light_pos[1] = (float)eye[Y];
	light_pos[2] = (float)eye[Z];
	light_pos[3] = 0.0f;
	glLightfv(GL_LIGHT0, GL_POSITION, light_pos);

    // 画面消去．
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);  

     // 外枠の描画．
    glColor3f(1.0f, 1.0f, 1.0f);
    glutWireCube(2.0);

    // 粒子群の描画．
    if (particle_mode)
        drawParticles();

	// マーチングキューブ法によるポリゴン群の描画．
	else {
		glEnable(GL_LIGHTING);
		glEnable(GL_NORMALIZE);
		marchingCubesIsoSurface();
		drawIsoSurface();
		glDisable(GL_LIGHTING);
	}
	
	// 粒子の移動．
	if (!pause_mode)
		moveParticles();
    glutSwapBuffers();
}

// リサイズ．
void resize(int width, int height)
{
    window_width = width;
    window_height = height;
}

// CUDA関係の初期化．
void initCUDA(void)
{

	// 格子構造に関するパラメータの転送．
	setParameters(&h_params);

	// メタボール法で用いるデバイスメモリの確保．
    cudaMalloc((void**)&d_position, num_particles * sizeof(float4));
    cudaMalloc((void**)&d_sorted_point, num_particles * sizeof(float4));
	cudaMalloc((void**)&d_grid_point_address, num_particles * sizeof(uint));
    cudaMalloc((void**)&d_grid_point_index, num_particles * sizeof(uint));
    cudaMalloc((void**)&d_voxel_start, num_voxels * sizeof(uint));
    cudaMalloc((void**)&d_voxel_end, num_voxels * sizeof(uint));
	cudaMalloc((void**)&d_density, num_voxels * sizeof(float));

    // マーチングキューブ法で用いるデバイスメモリの確保．
    cudaMalloc((void**)&d_num_vertices_table, 256 * sizeof(uint));
    cudaMalloc((void**)&d_triangle_table, 256 * 16 * sizeof(uint));
    cudaMalloc((void**)&d_voxel_vertices, num_voxels * sizeof(uint));
    cudaMalloc((void**)&d_voxel_vertices_scanned, num_voxels * sizeof(uint));
    cudaMalloc((void**)&d_voxel_occupied, num_voxels * sizeof(uint));
    cudaMalloc((void**)&d_voxel_occupied_scanned, num_voxels * sizeof(uint));
    cudaMalloc((void**)&d_compacted_voxel, num_voxels * sizeof(uint));

	// テクスチャメモリへのバインド．
	bindDensityTexture(d_density);
	bindNumVerticesTableTexture(d_num_vertices_table);
	bindTriangleTableTexture(d_triangle_table);
}

// 後処理．
void cleanUp(void)
{

    // メタボール法で用いたデバイスメモリの解放．
	cudaFree(d_position);
    cudaFree(d_sorted_point);
    cudaFree(d_grid_point_address);
    cudaFree(d_grid_point_index);
    cudaFree(d_voxel_start);
    cudaFree(d_voxel_end);
	cudaFree(d_density);

    // 頂点バッファオブジェクトの消去．
    deleteVBO(&poly_pos_vbo, poly_pos_vbo_res);
	deleteVBO(&poly_normal_vbo, poly_normal_vbo_res);

    // マーチングキューブ法で用いたデバイスメモリの解放．
    cudaFree(d_num_vertices_table);
    cudaFree(d_triangle_table);
    cudaFree(d_voxel_vertices);
    cudaFree(d_voxel_vertices_scanned);
    cudaFree(d_voxel_occupied);
    cudaFree(d_voxel_occupied_scanned);
    cudaFree(d_compacted_voxel);

	// CUDAテクスチャのアンバインド．
	unbindTextures();
	cudaDeviceReset();
}

// キー処理．
void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
    switch (key) {
        case 'q':
        case 'Q':
		case '\033':
            exit(0);
            break;

			// 粒子表示とポリゴン表示の切り替え．
        case 'p':
        case 'P':
            particle_mode = !particle_mode;
            break;

			// 一時停止．
        case ' ':
            pause_mode = !pause_mode;
            break;
    }
    glutPostRedisplay();
}

// マウスのボタン処理．
void mouse_button(int button, int state, int x, int y)
{
	if ((state == GLUT_DOWN) && (button == GLUT_LEFT_BUTTON))
		motion_p = true;
	else if (state == GLUT_UP)
		motion_p = false;
	mouse_old_x = x;
	mouse_old_y = y;
}

// マウスの移動処理．
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

// アイドル．
void idle(void)
{
     glutPostRedisplay();
}

// OpenGL関係の初期設定．
bool initGL(void)
{  
	glClearColor(0.25f, 0.25f, 0.25f, 1.0f);
    glEnable(GL_DEPTH_TEST);
	glClearDepth(1.0);
	glDepthFunc(GL_LESS);
	glEnable(GL_LIGHT0);

	// glewの初期化．     
    glewInit();
    if (!glewIsSupported("GL_VERSION_2_0")) {
		fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        return false;
    }

	// マーチングキューブ法で用いる頂点バッファオブジェクトの生成．
	createVBO(&poly_pos_vbo, max_num_vertices * 4 * sizeof(float), &poly_pos_vbo_res, cudaGraphicsRegisterFlagsWriteDiscard);
	createVBO(&poly_normal_vbo, max_num_vertices * 4 * sizeof(float), &poly_normal_vbo_res, cudaGraphicsRegisterFlagsWriteDiscard);
	return true;
}

int main(int argc, char** argv) 
{

    // セルの格子の解像度とサイズ．
    num_voxels = grid_size.x * grid_size.y * grid_size.z;
    printf("grid: %d x %d x %d = %d voxels\n", grid_size.x, grid_size.y, grid_size.z, num_voxels);
    voxel_size = make_float4(world_size.x / grid_size.x, world_size.y / grid_size.y, world_size.z / grid_size.z, 1.0f);

    // シミュレーションパラメータの保存．
    h_params.grid_size = grid_size;
    h_params.num_voxels = num_voxels;
    h_params.world_origin = world_origin; 
    h_params.voxel_size = voxel_size;

    // 想定する最大頂点数．
    max_num_vertices = grid_size.x * grid_size.y * 100;
    printf("max num vertices = %d\n", max_num_vertices);

	// CUDA関係での必要なメモリの確保．
	initCUDA();

	// 粒子を初期位置に設定．
	setInitialPosition();

    // GLUTの定義．
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

	// OpenGLの設定．
	if (!initGL())
        return 1;
	
    // アニメーション描画のループ．
    glutMainLoop();
    return 0;
}