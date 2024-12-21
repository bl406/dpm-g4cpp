#ifndef _3DTF_H_
#define _3DTF_H_

#ifndef M_PI
    #define M_PI 3.14159265358979323846f
#endif

enum Axis {
    X, Y, Z,
    ZXZ, ZYZ
};
enum Rotation {
    INTRINSIC, EXTRINSIC
};

struct Vector3D;
struct Matrix3D;
struct EluerAngle;

/// @brief Vector3D
struct Vector3D {
    float x;
    float y;
    float z;

public:
    __device__ __host__ Vector3D();
    __device__ __host__ Vector3D(const float& x, const float& y, const float& z);
    int print();
    __device__ __host__
    int getCoordinate(float& x, float& y, float& z);
    __device__ __host__
    float operator[](int i) const;
};

/// @brief 3X3 Matrix
struct Matrix3D {
    float m[3][3];

public:
    __device__ __host__
    Matrix3D();
    __device__ __host__
    Matrix3D(const float& angle, const Axis& axis);
};

/// @brief eluer angle
struct EluerAngle {
    float alpha;
    float beta;
    float gamma;

    Axis axis;
    Rotation rotation;
    
    Matrix3D m; // rotation matrix
    __device__ __host__
    Matrix3D initMatrix();

public:
    __device__ __host__ EluerAngle();
    __device__ __host__ EluerAngle(const float& alpha, const float& beta, const float& gamma, const Axis& axis, const Rotation& rotation);
    __device__ __host__ Matrix3D getMatrix() const;
};

__device__ __host__
Vector3D operator+(const Vector3D& a, const Vector3D& b);
__device__ __host__
Vector3D operator-(const Vector3D& a, const Vector3D& b);
__device__ __host__
Vector3D operator*(const float& c, const Vector3D& b);
__device__ __host__
Vector3D operator*(const Matrix3D& m, const Vector3D& v);
__device__ __host__
Matrix3D operator*(const Matrix3D& a, const Matrix3D& b);

/* translate vector */
__device__ __host__
Vector3D translate(const Vector3D& v, const Vector3D& delta);

/* rotate vector */
__device__ __host__
Vector3D rotate(const Vector3D& v, const Matrix3D& m);
__device__ __host__
Vector3D rotate(const Vector3D& v, const EluerAngle& eluerAngle);

#endif  // _3DTF_H_
