#include <stdio.h>
#include <math.h>

#include "BasicDef.h"

// 2�{�̃x�N�g��vec0��vec1�̓��ρD
double dot(double vec0[], double vec1[])
// double vec0[];
// double vec1[];
{
	return(vec0[X] * vec1[X] + vec0[Y] * vec1[Y] + vec0[Z] * vec1[Z]);
}

// 2�{�̃x�N�g��vec0��vec1�̊O�ρD
void cross(double vec0[], double vec1[], double vec2[])
// double vec0[];
// double vec1[];
// double vec2[]; vec0 X vec1.
{
	vec2[X] = vec0[Y] * vec1[Z] - vec0[Z] * vec1[Y];
	vec2[Y] = vec0[Z] * vec1[X] - vec0[X] * vec1[Z];
	vec2[Z] = vec0[X] * vec1[Y] - vec0[Y] * vec1[X];
}

// �x�N�g���̐��K���D
void normVec(double vec[])
// double vec[]; ���ӁI���̃x�N�g���͔j��I�ɕύX�����D
{
	double norm;
	norm = sqrt(vec[X] * vec[X] + vec[Y] * vec[Y] + vec[Z] * vec[Z]);
	vec[X] /= norm;
	vec[Y] /= norm;
	vec[Z] /= norm;
}

// 3���_���܂ޕ��ʂ̒P�ʖ@���x�N�g���̌v�Z�D3���_�������v����ɕ���ł��邱
// �Ƃ�����D
void normal(double p0[], double p1[], double p2[], double normal[])
// double p0[], p1[], p2[]; 3���_�̍��W�D
// double normal[]; �v�Z���ꂽ�@���x�N�g���D
{
	unsigned int i;
	double v0[3], v1[3];

	// ��{�ƂȂ�Q�̃x�N�g���𐶐��D
	for (i = 0; i < 3; i++) {
		v0[i] = p2[i] - p1[i];
		v1[i] = p0[i] - p1[i];
	}

	// ���������x�N�g���̊O�ς��v�Z����D
	cross(v0, v1, normal);

	// �O�ςɂ���ē���ꂽ�@���x�N�g���𐳋K���D
	normVec(normal);
}

// �@�������ƒʉߓ_�̎w�肩�畽�ʂ̕�����������D
void defPlane(double normal[], double point[], double plane_eq[])
// double normal[]; ���ʂ̖@���x�N�g���D
// double point[]; ���ʏ��1�_�̍��W�D
// double plane_eq[]; ���ʂ̕�����ax + by + cz + d = 0�̌W��[a, b, c, d]�D
{
	plane_eq[A] = normal[X];
	plane_eq[B] = normal[Y];
	plane_eq[C] = normal[Z];
	plane_eq[D] = - (normal[X] * point[X] + normal[Y] * point[Y] + normal[Z] * point[Z]);
}

// �����ƕ��ʂ̌�_�D
void intPointLinePlane(double dir[], double point[], double plane_eq[], double int_point[])
// double dir[]; �����̐��K�����ꂽ�����x�N�g���D
// double point[]; �������1�_�̍��W�D
// double plane_eq[]; ���ʂ̕������D�����ƕ��s�ł͂Ȃ����Ƃ�����D
// double int_point[]; ��_�̍��W�D
{
	double d, t;
	d = dot(dir, plane_eq);
	t = (- (plane_eq[A] * point[X] + plane_eq[B] * point[Y] + plane_eq[C] * point[Z] + plane_eq[D])) / d;
	int_point[X] = dir[X] * t + point[X];
	int_point[Y] = dir[Y] * t + point[Y];
	int_point[Z] = dir[Z] * t + point[Z];
}
