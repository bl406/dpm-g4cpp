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

    /* Step 1:计算出粒子在绝对坐标系下的位置和方向 */
    /* 粒子在绝对坐标系下的位置方向可由粒子在粒子源坐标系下的位置方向加上粒子源坐标系到绝对坐标系的转换叠加而成.
    粒子源坐标系相对于绝对坐标系的位置确定需要三步完成   
    Step 1.1,内旋zyz顺规旋转;
    Step 1.2,isocenter平移.
    */
    Vector3D position(x, y, z); // 粒子在粒子源坐标系下的位置向量
    Vector3D direction(u, v, w); // 粒子在粒子源坐标系下的速度方向向量

    // Step 1.1,完成内旋zyz顺规旋转
    EluerAngle eluerAngle(phi, theta, phicol, ZYZ, INTRINSIC);
    position = rotate(position, eluerAngle);
    direction = rotate(direction, eluerAngle);

    // Step 1.2,完成isocenter参数对坐标系的平移
    Vector3D isocenterV3D(xiso, yiso, ziso);
    position = translate(position, isocenterV3D);

    x = position.x;
    y = position.y;
    z = position.z;
    u = direction.x;
    v = direction.y;
    w = direction.z;

    /* Step 2:确定粒子在模体的初始落点 */
    /* 使用图形学光线追踪理论中的AABB包围盒方法求得粒子与模体关系并求得粒子在模体的初始region*/
    Vector3D aabbMin = Vector3D(Geometry::d_Xbound[0], Geometry::d_Ybound[0], Geometry::d_Zbound[0]);
	Vector3D aabbMax = Vector3D(Geometry::d_Xbound[1], Geometry::d_Ybound[1], Geometry::d_Zbound[1]);

    float tEnterMAX, tExitMIN;
    // 一定要初始化为负无穷否则会出现错误
    tEnterMAX = -FLT_MAX;
    tExitMIN = FLT_MAX;
    float tEnter, tExit;
    for (int i = 0; i < 3; i++) {
        if (direction[i] == 0.0) {
            // 永远不可能与模体相交
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

    // 粒子在模体内
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
    // 粒子在模体外,且粒子经过模体
    else if (0 < tEnterMAX && tEnterMAX < tExitMIN) {
        /*
        插眼: 量化问题
        x = position.x + tEnterMAX * direction.x;
        tEnterMAX = (aabbMin.x - position.x) / direction.x;
        从???
        x = position.x + (aabbMin.x - position.x) / direction.x * direction.x;
        在这个过程中,如果x约定???,但int(x-0)=-1,那么就会对ix的值造成影响,从而使程序崩溃.
        */
        x += tEnterMAX * direction.x;
        y += tEnterMAX * direction.y;
        z += tEnterMAX * direction.z;

        ix = (int)((x - Geometry::d_Xbound[0]) / Geometry::d_Spacing[0]);
        iy = (int)((y - Geometry::d_Ybound[0]) / Geometry::d_Spacing[1]);
        iz = (int)((z - Geometry::d_Zbound[0]) / Geometry::d_Spacing[2]);

        /*
        由于geometry是离散的,而x,y,z近似是连续的.
        因此geometry内部的ix可由int或者ceil得到.
        但是geometry表面的位???x与geometry-GetXbound()[0]数值上绝对相等),则需要特殊处???所有表面位置的粒子都归为geometry内部.
        因此需要对ix,iy,iz进行特殊处理.
        向下取整int,则需要对ix==Geometry::d_Dim[0]的情况进行特殊处???
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
    // 粒子不经过模体
    else {
        ix = iy = iz = -1;
    }
}