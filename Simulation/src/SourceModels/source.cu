/**
 * @file source.cpp
 * @brief Source class implementation
 */

#include "source.h"
#include "collipnt_source.h"
#include "cone_source.h"
#include "3dtf.hpp"

#include "math.h"
#include "string.h"
#include "stdlib.h"
#include "float.h"
#include "omc_utilities.h"
#include "geom.hh"
#include "Track.hh"

class Source *source;

Source* initSource(Geom* geom, std::string srcFn) {
    char buffer[BUFFER_SIZE];

	parseInputFile((char*)srcFn.c_str());

    /* First get isource */
    if (getInputValue(buffer, "isource") != 1) {
        printf("Can not find 'isource' key on input file.\n");
        exit(EXIT_FAILURE);
    }
    SourceType isource = (SourceType)atoi(buffer);

    switch (isource) {
        case COLLIPOINT_SOURCE:
            source = new CollipntSource(geom);
        break;
        case CONE_SOURCE:
			//source = new ConeSource(geom);
			break;
         default:
             printf("isource = %d is not supported.\n", isource);
             exit(EXIT_FAILURE);
    }
   
    return source;
}

void cleanSource() {
    delete source;

    return;
}

__device__ void cal_entrance_point(float phi, float theta, float phicol,
    float xiso, float yiso, float ziso,
    float& x, float& y, float& z,
	int& ix, int& iy, int& iz,
    float& u, float& v, float& w) {

    /* Step 1:����������ھ�������ϵ�µ�λ�úͷ��� */
    /* �����ھ�������ϵ�µ�λ�÷����������������Դ����ϵ�µ�λ�÷����������Դ����ϵ����������ϵ��ת�����Ӷ���.
    ����Դ����ϵ����ھ�������ϵ��λ��ȷ����Ҫ�������   
    Step 1.1,����zyz˳����ת;
    Step 1.2,isocenterƽ��.
    */
    Vector3D position(x, y, z); // ����������Դ����ϵ�µ�λ������
    Vector3D direction(u, v, w); // ����������Դ����ϵ�µ��ٶȷ�������

    // Step 1.1,�������zyz˳����ת
    EluerAngle eluerAngle(phi, theta, phicol, ZYZ, INTRINSIC);
    position = rotate(position, eluerAngle);
    direction = rotate(direction, eluerAngle);

    // Step 1.2,���isocenter����������ϵ��ƽ��
    Vector3D isocenterV3D(xiso, yiso, ziso);
    position = translate(position, isocenterV3D);

    x = position.x;
    y = position.y;
    z = position.z;
    u = direction.x;
    v = direction.y;
    w = direction.z;

    /* Step 2:ȷ��������ģ��ĳ�ʼ��� */
    /* ʹ��ͼ��ѧ����׷�������е�AABB��Χ�з������������ģ���ϵ�����������ģ��ĳ�ʼregion*/
    Vector3D aabbMin = Vector3D(Geometry::d_Xbound[0], Geometry::d_Ybound[0], Geometry::d_Zbound[0]);
	Vector3D aabbMax = Vector3D(Geometry::d_Xbound[1], Geometry::d_Ybound[1], Geometry::d_Zbound[1]);

    float tEnterMAX, tExitMIN;
    // һ��Ҫ��ʼ��Ϊ������������ִ���
    tEnterMAX = -FLT_MAX;
    tExitMIN = FLT_MAX;
    float tEnter, tExit;
    for (int i = 0; i < 3; i++) {
        if (direction[i] == 0.0) {
            // ��Զ��������ģ���ཻ
            if (position[i] < aabbMin[i] || position[i] > aabbMax[i]) {
                tEnter = FLT_MAX;
                tExit = -FLT_MAX;
            }
            else {
                tEnter = -FLT_MAX;
                tExit = FLT_MAX;
            }
        }
        else if (direction[i] > 0.0) {
            tEnter = (aabbMin[i] - position[i]) / direction[i];
            tExit = (aabbMax[i] - position[i]) / direction[i];
        }
        else {
            tEnter = (aabbMax[i] - position[i]) / direction[i];
            tExit = (aabbMin[i] - position[i]) / direction[i];
        }

        if (tEnter > tEnterMAX) {
            tEnterMAX = tEnter;
        }
        if (tExit < tExitMIN) {
            tExitMIN = tExit;
        }
    }

    // ������ģ����
    if (tEnterMAX <= 0 && 0 <= tExitMIN) {
        ix = (int)((x - Geometry::d_Xbound[0]) / Geometry::d_Spacing[0]);
        iy = (int)((y - Geometry::d_Ybound[0]) / Geometry::d_Spacing[1]);
        iz = (int)((z - Geometry::d_Zbound[0]) / Geometry::d_Spacing[2]);

        if (ix == Geometry::d_Dim[0]) {
            ix = Geometry::d_Dim[0] - 1;
        }
        if (iy == Geometry::d_Dim[1]) {
            iy = Geometry::d_Dim[1] - 1;
        }
        if (iz == Geometry::d_Dim[2]) {
            iz = Geometry::d_Dim[2] - 1;
        }       
    }
    // ������ģ����,�����Ӿ���ģ��
    else if (0 < tEnterMAX && tEnterMAX < tExitMIN) {
        /*
        ����: ��������
        x = position.x + tEnterMAX * direction.x;
        tEnterMAX = (aabbMin.x - position.x) / direction.x;
        ��???
        x = position.x + (aabbMin.x - position.x) / direction.x * direction.x;
        �����������,���xԼ��???,��int(x-0)=-1,��ô�ͻ��ix��ֵ���Ӱ��,�Ӷ�ʹ�������.
        */
        x += tEnterMAX * direction.x;
        y += tEnterMAX * direction.y;
        z += tEnterMAX * direction.z;

        ix = (int)((x - Geometry::d_Xbound[0]) / Geometry::d_Spacing[0]);
        iy = (int)((y - Geometry::d_Ybound[0]) / Geometry::d_Spacing[1]);
        iz = (int)((z - Geometry::d_Zbound[0]) / Geometry::d_Spacing[2]);

        /*
        ����geometry����ɢ��,��x,y,z������������.
        ���geometry�ڲ���ix����int����ceil�õ�.
        ����geometry�����λ???x��geometry-GetXbound()[0]��ֵ�Ͼ������),����Ҫ���⴦???���б���λ�õ����Ӷ���Ϊgeometry�ڲ�.
        �����Ҫ��ix,iy,iz�������⴦��.
        ����ȡ��int,����Ҫ��ix==Geometry::d_Dim[0]������������⴦???
        */
        if (ix == Geometry::d_Dim[0]) {
            ix = Geometry::d_Dim[0] - 1;
        }
        if (iy == Geometry::d_Dim[1]) {
            iy = Geometry::d_Dim[1] - 1;
        }
        if (iz == Geometry::d_Dim[2]) {
            iz = Geometry::d_Dim[2] - 1;
        }
    }
    // ���Ӳ�����ģ��
    else {
        ix = iy = iz = -1;
    }
}