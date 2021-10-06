#include <stdio.h>
// #include <gl/freeglut.h>

#include <GL/glut.h>
#include <math.h>

#include "BasicDef.h"

// ���_�ݒ�Ɋւ���ϐ��D
double eye[3];
double center[3] = {0.0, 0.0, 0.0};
double up[3];

// �����e�̍��W�n�D
double x_axis[3], y_axis[3], z_axis[3];

// �E�B���h�E��̍��W�ƃ��f�����W�̕ϊ��ɗp����f�[�^�D
double corner[3];
double pixel_size, ref_far, ref_near;

// �Ɩ��D
GLfloat light_pos[4];

// �_�D
extern double point[MAX_NUM_POINTS][3];
extern unsigned int num_points;

// �O�p�`�|���S���D
extern unsigned int triangle[MAX_NUM_TRIANGLES][3];
extern unsigned int num_triangles;

// ��{�I�Ȑ}�`�v�Z�D
extern double dot(double vec0[], double vec1[]);
extern void cross(double vec0[], double vec1[], double vec2[]);
extern void normVec(double vec[]);
extern void normal(double p0[], double p1[], double p2[], double normal[]);

// ���e�����p�̍s���`�D���f�����E�B���h�E��t�ɕ\������D
void defineViewMatrix(double phi, double theta, unsigned int width, unsigned int height, double margin)
// double phi, theta; ���ʊp�Ƌp.
// unsigned int width, height; �E�B���h�E�̉𑜓x�D
// double margin; �\������ۂɗ��̎��͂ɗ^����]�T�D
{
	unsigned int i, j;
	double c, s, xy_dist;
	double vec[3];
	double left, right, bottom, top, farVal, nearVal;
	double dx, dy, d_aspect, w_aspect, d;

	// ���_�̐ݒ�D
	eye[Z] = sin(theta * PI / 180.0);
	xy_dist = cos(theta * PI / 180.0);
	c = cos(phi * PI / 180.0);
	s = sin(phi * PI / 180.0);
	eye[X] = xy_dist * c;
	eye[Y] = xy_dist * s;
	up[X] = - c * eye[Z];
	up[Y] = - s * eye[Z];
	up[Z] = s * eye[Y] + c * eye[X];
	normVec(up);

	// ���_�����_�Ƃ�����W�n�̒�`�D
	for (i = 0; i < 3; i++)
		z_axis[i] = eye[i] - center[i];
	normVec(z_axis);
	cross(up, z_axis, x_axis);
	normVec(x_axis);
	cross(z_axis, x_axis, y_axis);

	// left, right, bottom, top, nearVal, farVal�̌���D
	left = bottom = farVal = LARGE;
	right = top = nearVal = - LARGE;
	for (i = 0; i < num_points; i++) {
		for (j = 0; j < 3; j++)
			vec[j] = point[i][j] - eye[j];		
		if (dot(x_axis, vec) < left)
			left = dot(x_axis, vec);
		if (dot(x_axis, vec) > right)
			right = dot(x_axis, vec);
		if (dot(y_axis, vec) < bottom)
			bottom = dot(y_axis, vec);
		if (dot(y_axis, vec) > top)
			top = dot(y_axis, vec);
		if (dot(z_axis, vec) < farVal)
			farVal = dot(z_axis, vec);
		if (dot(z_axis, vec) > nearVal)
			nearVal = dot(z_axis, vec);
	}

	// �}�`�̎��͂�5%�قǗ]�T��^����D
	margin += (right - left) * 0.05;
	left -= margin;
	right += margin;
	bottom -= margin;
	top += margin;
	farVal -= margin;
	nearVal += margin;
	ref_far = farVal;
	ref_near = nearVal;

	// �\���͈͂̃A�X�y�N�g��ƃE�B���h�E�̃A�X�y�N�g��̔�r�D
	dx = right - left;
	dy = top - bottom;
	d_aspect = dy / dx;
	w_aspect = (double)height / (double)width;

	// �E�B���h�E���\���͈͂����c���D�\���͈͂��c�ɍL����D
	if (w_aspect > d_aspect) {
		d = (dy * (w_aspect / d_aspect - 1.0)) * 0.5;
		bottom -= d;
		top += d;

		// �E�B���h�E���\���͈͂��������D�\���͈͂����ɍL����D
	} else {
		d = (dx * (d_aspect / w_aspect - 1.0)) * 0.5;
		left -= d;
		right += d;
	}

	// �e�s�N�Z���̎��T�C�Y�D
	pixel_size = (right - left) / width;

	// ���̐ς̍��������̍��W���L�^���Ă����D
	for (i = 0; i < 3; i++)
		corner[i] = eye[i] + bottom * y_axis[i] + left * x_axis[i] + farVal * z_axis[i];

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(left, right, bottom, top, - nearVal, - farVal);
	glViewport(0, 0, width, height);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(eye[X], eye[Y], eye[Z], center[X], center[Y], center[Z], up[X], up[Y], up[Z]); 
}

// ���f���\���֐��D
void displayModel(void)
{
	unsigned int i;
	double nrml_vec[3];

	// �����̐ݒ�D
	light_pos[0] = (float)eye[X];
	light_pos[1] = (float)eye[Y];
	light_pos[2] = (float)eye[Z];
	light_pos[3] = 0.0f;
	glLightfv(GL_LIGHT0, GL_POSITION, light_pos);

	// �Ɩ��̓_���D
	glEnable(GL_LIGHTING);

	// �`�揈���D
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glBegin(GL_TRIANGLES);
	for (i = 0; i < num_triangles; i++) {
		normal(point[triangle[i][0]], point[triangle[i][1]], point[triangle[i][2]], nrml_vec);
		if (dot(nrml_vec, z_axis) > 0.0)
			glNormal3dv(nrml_vec);
		else
			glNormal3d(- nrml_vec[X], - nrml_vec[Y], - nrml_vec[Z]);
		glVertex3dv(point[triangle[i][0]]);
		glVertex3dv(point[triangle[i][1]]);
		glVertex3dv(point[triangle[i][2]]);
	}
	glEnd();
	glFlush();
}

// �s�N�Z���̉������Əc�����̔Ԓn(h, v)�ƃf�v�X�ld����C�Ή����郂�f�����W�𓾂�D
void pixelCoordToModelCoord(int h, int v, float d, double point[])
// int h, v; �s�N�Z���̉������Əc�����̔Ԓn�D
// float d; �f�v�X�l�D
// double point[]; �Ή����郂�f�����W�D
{
	unsigned int i;

	// corner�ɂ͍��������ɑΉ����郂�f�����W���i�[����Ă���D
	for (i = 0; i < 3; i++) {
		point[i] = corner[i] + pixel_size * h * x_axis[i] + pixel_size * v * y_axis[i]

			// �f�v�X�l��near��0.0�Cfar��1.0�ɑΉ�����悤�ɐ��K�����Ă���D
			+ (ref_near - ref_far) * (1.0f - d) * z_axis[i];
	}
}

// ���f�����W����Ή�����E�B���h�E��̔Ԓn�𓾂�D
void modelCoordToPixelCoord(double point[], int *h_index, int *v_index)
// double point[]; ���f�����W�D
// int *h_index, *v_index; �Ή�����s�N�Z���̔Ԓn�D
{
	double diff[3];
	diff[X] = point[X] - corner[X];
	diff[Y] = point[Y] - corner[Y];
	diff[Z] = point[Z] - corner[Z];
	*h_index = (int)floor(dot(diff, x_axis) / pixel_size);
	*v_index = (int)floor(dot(diff, y_axis) / pixel_size);
}
