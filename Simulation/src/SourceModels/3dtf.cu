#include "3dtf.hpp"

#include <cstdio>
#include <cmath>

__device__ __host__
Vector3D::Vector3D() { }

__device__ __host__
Vector3D::Vector3D(const float& x, const float& y, const float& z)
{
    this->x = x;
    this->y = y;
    this->z = z;
}

int Vector3D::print()
{
    return printf("(%f, %f, %f)", this->x, this->y, this->z);
}

__device__ __host__
int Vector3D::getCoordinate(float& x, float& y, float& z)
{
    x = this->x;
    y = this->y;
    z = this->z;
    return 1;
}

__device__ __host__
float Vector3D::operator[](int i) const
{
    switch (i) {
        case 0:
            return this->x;
        case 1:
            return this->y;
        case 2:
        default:
            return this->z;
    }
}

__device__ __host__
Matrix3D::Matrix3D() { }

__device__ __host__
Matrix3D::Matrix3D(const float& angle, const Axis& axis)
{
    float c = cos(angle * M_PI / 180.0f);
    float s = sin(angle * M_PI / 180.0f);

    switch (axis) {
        case X:
            this->m[0][0] = 1.0;
            this->m[0][1] = 0.0;
            this->m[0][2] = 0.0;
            this->m[1][0] = 0.0;
            this->m[1][1] = c;
            this->m[1][2] = -s;
            this->m[2][0] = 0.0;
            this->m[2][1] = s;
            this->m[2][2] = c;
            break;

        case Y:
            this->m[0][0] = c;
            this->m[0][1] = 0.0;
            this->m[0][2] = s;
            this->m[1][0] = 0.0;
            this->m[1][1] = 1.0;
            this->m[1][2] = 0.0;
            this->m[2][0] = -s;
            this->m[2][1] = 0.0;
            this->m[2][2] = c;
            break;

        case Z:
        default:
            this->m[0][0] = c;
            this->m[0][1] = -s;
            this->m[0][2] = 0.0;
            this->m[1][0] = s;
            this->m[1][1] = c;
            this->m[1][2] = 0.0;
            this->m[2][0] = 0.0;
            this->m[2][1] = 0.0;
            this->m[2][2] = 1.0;
            break;              
    }
}

__device__ __host__
EluerAngle::EluerAngle() { }

__device__ __host__
EluerAngle::EluerAngle(const float& alpha, const float& beta, const float& gamma, const Axis& axis, const Rotation& rotation)
{
    this->alpha = alpha;
    this->beta = beta;
    this->gamma = gamma;
    this->axis = axis;
    this->rotation = rotation;
    this->m = initMatrix();
}

__device__ __host__
Matrix3D EluerAngle::initMatrix()
{
    switch (this->axis) {
        case ZYZ:
            if (this->rotation == EXTRINSIC) {
                return Matrix3D(this->gamma, Z) * Matrix3D(this->beta, Y) * Matrix3D(this->alpha, Z);
            } else {
                return Matrix3D(this->alpha, Z) * Matrix3D(this->beta, Y) * Matrix3D(this->gamma, Z);
            }
            break;

        default:
        case ZXZ:
            if (this->rotation == EXTRINSIC) {
                return Matrix3D(this->gamma, Z) * Matrix3D(this->beta, X) * Matrix3D(this->alpha, Z);
            } else {
                return Matrix3D(this->alpha, Z) * Matrix3D(this->beta, X) * Matrix3D(this->gamma, Z);
            }
            break;
    }
}

__device__ __host__
Matrix3D EluerAngle::getMatrix() const
{
    return this->m;
}

__device__ __host__
Vector3D operator+(const Vector3D& a, const Vector3D& b)
{
    return Vector3D(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __host__
Vector3D operator-(const Vector3D& a, const Vector3D& b)
{
    return Vector3D(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __host__
Vector3D operator*(const float& c, const Vector3D& b)
{
    return Vector3D(c * b.x, c * b.y, c * b.z);
}

__device__ __host__
Vector3D operator*(const Matrix3D& m,const Vector3D& v)
{
    return Vector3D(m.m[0][0]*v.x + m.m[0][1]*v.y + m.m[0][2]*v.z,
                    m.m[1][0]*v.x + m.m[1][1]*v.y + m.m[1][2]*v.z,
                    m.m[2][0]*v.x + m.m[2][1]*v.y + m.m[2][2]*v.z);
}

__device__ __host__
Matrix3D operator*(const Matrix3D& a, const Matrix3D& b)
{
    Matrix3D m;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            m.m[i][j] = 0.0;
            for (int k = 0; k < 3; k++) {
                m.m[i][j] += a.m[i][k] * b.m[k][j];
            }
        }
    }
    return m;
}

/// @brief translate vector
/// @param v original vector
/// @param delta translation vector
/// @return v + delta
__device__ __host__
Vector3D translate(const Vector3D& v, const Vector3D& delta)
{
    return v + delta;
}

/// @brief rotate vector
/// @param v original vector
/// @param m rotation matrix
/// @return rotated vector
__device__ __host__
Vector3D rotate(const Vector3D& v, const Matrix3D& m)
{
    return m * v;
}

/// @brief rotate vector
/// @param v original vector
/// @param eluerAngle eluer angle
/// @return rotated vector
__device__ __host__
Vector3D rotate(const Vector3D& v, const EluerAngle& eluerAngle)
{
    return eluerAngle.getMatrix() * v;
}
