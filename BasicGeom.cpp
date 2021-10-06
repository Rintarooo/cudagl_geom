#include <stdio.h>
#include <math.h>

#include "BasicDef.h"

// 2本のベクトルvec0とvec1の内積．
double dot(double vec0[], double vec1[])
// double vec0[];
// double vec1[];
{
	return(vec0[X] * vec1[X] + vec0[Y] * vec1[Y] + vec0[Z] * vec1[Z]);
}

// 2本のベクトルvec0とvec1の外積．
void cross(double vec0[], double vec1[], double vec2[])
// double vec0[];
// double vec1[];
// double vec2[]; vec0 X vec1.
{
	vec2[X] = vec0[Y] * vec1[Z] - vec0[Z] * vec1[Y];
	vec2[Y] = vec0[Z] * vec1[X] - vec0[X] * vec1[Z];
	vec2[Z] = vec0[X] * vec1[Y] - vec0[Y] * vec1[X];
}

// ベクトルの正規化．
void normVec(double vec[])
// double vec[]; 注意！このベクトルは破壊的に変更される．
{
	double norm;
	norm = sqrt(vec[X] * vec[X] + vec[Y] * vec[Y] + vec[Z] * vec[Z]);
	vec[X] /= norm;
	vec[Y] /= norm;
	vec[Z] /= norm;
}

// 3頂点を含む平面の単位法線ベクトルの計算．3頂点が反時計周りに並んでいるこ
// とを仮定．
void normal(double p0[], double p1[], double p2[], double normal[])
// double p0[], p1[], p2[]; 3頂点の座標．
// double normal[]; 計算された法線ベクトル．
{
	unsigned int i;
	double v0[3], v1[3];

	// 基本となる２つのベクトルを生成．
	for (i = 0; i < 3; i++) {
		v0[i] = p2[i] - p1[i];
		v1[i] = p0[i] - p1[i];
	}

	// 生成したベクトルの外積を計算する．
	cross(v0, v1, normal);

	// 外積によって得られた法線ベクトルを正規化．
	normVec(normal);
}

// 法線方向と通過点の指定から平面の方程式を決定．
void defPlane(double normal[], double point[], double plane_eq[])
// double normal[]; 平面の法線ベクトル．
// double point[]; 平面上の1点の座標．
// double plane_eq[]; 平面の方程式ax + by + cz + d = 0の係数[a, b, c, d]．
{
	plane_eq[A] = normal[X];
	plane_eq[B] = normal[Y];
	plane_eq[C] = normal[Z];
	plane_eq[D] = - (normal[X] * point[X] + normal[Y] * point[Y] + normal[Z] * point[Z]);
}

// 直線と平面の交点．
void intPointLinePlane(double dir[], double point[], double plane_eq[], double int_point[])
// double dir[]; 直線の正規化された方向ベクトル．
// double point[]; 直線上の1点の座標．
// double plane_eq[]; 平面の方程式．直線と平行ではないことを仮定．
// double int_point[]; 交点の座標．
{
	double d, t;
	d = dot(dir, plane_eq);
	t = (- (plane_eq[A] * point[X] + plane_eq[B] * point[Y] + plane_eq[C] * point[Z] + plane_eq[D])) / d;
	int_point[X] = dir[X] * t + point[X];
	int_point[Y] = dir[Y] * t + point[Y];
	int_point[Z] = dir[Z] * t + point[Z];
}
