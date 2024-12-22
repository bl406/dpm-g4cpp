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
#include "Random.hh"

class Source *source;

namespace Spectrum {
    __constant__ int d_deltak;
	__constant__ int d_srctype;
    __constant__ float d_cdfinv1[INVDIM];
    __constant__ float d_cdfinv2[INVDIM];
	__constant__ float d_energy;

    __device__ float getEkin(){
        /* Get primary particle energy */
        float ein = 0.0f;
        switch (d_srctype) {
        case CollipntSource::MONOENERGETIC:
        default:
            ein = d_energy;
            break;
        case CollipntSource::SPECTRUM:
            /* Sample initial energy from spectrum data */
            float rnno1 = CuRand::rand();
            float rnno2 = CuRand::rand();

            /* Sample bin number in order to select particle energy */
            int k = (int)fmin(d_deltak * rnno1, d_deltak - 1.0f);
            ein = d_cdfinv1[k] + rnno2 * d_cdfinv2[k];
            break;          
        }
		return ein;
    }
}

Source* initSource(Geom* geom, std::string srcFn) {
    char buffer[BUFFER_SIZE];

	parseInputFile((char*)srcFn.c_str());

    /* First get isource */
    if (getInputValue(buffer, "isource") != 1) {
        printf("Can not find 'isource' key on input file.\n");
        exit(EXIT_FAILURE);
    }
    Source::SourceType isource = (Source::SourceType)atoi(buffer);

    switch (isource) {
        case Source::COLLIPOINT_SOURCE:
            source = new CollipntSource(geom);
        break;
        case Source::CONE_SOURCE:
			//source = new ConeSource(geom);
			break;
         default:
             printf("isource = %d is not supported.\n", isource);
             exit(EXIT_FAILURE);
    }
   
    return source;
}

Source::Source(Geom* geom) {
	this->geometry = geom;

    char buffer[BUFFER_SIZE];

    /* Get source file path from input data */
    source_type = SPECTRUM;    /* energy spectrum as default case */
    energy = 6.f;
    /* First check of spectrum file was given as an input */
    if (getInputValue(buffer, "spectrum file") != 1) {
        printf("Can not find 'spectrum file' key on input file.\n");
        printf("Switch to monoenergetic case.\n");
        source_type = MONOENERGETIC;    /* monoenergetic source */
    }

    switch (source_type) {
    case MONOENERGETIC:
        if (getInputValue(buffer, "mono energy") != 1) {
            printf("Can not find 'mono energy' key on input file.\n");
            exit(EXIT_FAILURE);
        }
        energy = (float)atof(buffer);
        printf("%f monoenergetic source\n", energy);
        break;

    case SPECTRUM:
        char spectrum_file[128];    // spectrum file path
        removeSpaces(spectrum_file, buffer);

        /* Open .source file */
        FILE* fp;

        if ((fp = fopen(spectrum_file, "r")) == NULL) {
            printf("Unable to open file: %s\n", spectrum_file);
            exit(EXIT_FAILURE);
        }

        printf("Path to spectrum file : %s\n", spectrum_file);

        /* Read spectrum file title */
        fgets(buffer, BUFFER_SIZE, fp);
        printf("Spectrum file title: %s", buffer);

        /* Read number of bins and spectrum type */
        float enmin;   /* lower energy of first bin */
        int nensrc;     /* number of energy bins in spectrum histogram */
        int imode;      /* 0 : histogram counts/bin, 1 : counts/MeV*/

        fgets(buffer, BUFFER_SIZE, fp);
        sscanf(buffer, "%d %f %d", &nensrc, &enmin, &imode);

        if (nensrc > MXEBIN) {
            printf("Number of energy bins = %d is greater than max allowed = "
                "%d. Increase MXEBIN macro!\n", nensrc, MXEBIN);
            exit(EXIT_FAILURE);
        }

        /* upper energy of bin i in MeV */
        float* ensrcd = (float*)malloc(nensrc * sizeof(float));
        /* prob. of finding a particle in bin i */
        float* srcpdf = (float*)malloc(nensrc * sizeof(float));

        /* Read spectrum information */
        for (int i = 0; i < nensrc; i++) {
            fgets(buffer, BUFFER_SIZE, fp);
            sscanf(buffer, "%f %f", &ensrcd[i], &srcpdf[i]);
        }
        printf("Have read %d input energy bins from spectrum file.\n", nensrc);

        if (imode == 0) {
            printf("Counts/bin assumed.\n");
        }
        else if (imode == 1) {
            printf("Counts/MeV assumed.\n");
            srcpdf[0] *= (ensrcd[0] - enmin);
            for (int i = 1; i < nensrc; i++) {
                srcpdf[i] *= (ensrcd[i] - ensrcd[i - 1]);
            }
        }
        else {
            printf("Invalid mode number in spectrum file.");
            exit(EXIT_FAILURE);
        }

        float ein = ensrcd[nensrc - 1];
        printf("Energy ranges from %f to %f MeV\n", enmin, ein);

        /* Initialization routine to calculate the inverse of the
        cumulative probability distribution that is used during execution to
        sample the incident particle energy. */
        float* srccdf = (float*)malloc(nensrc * sizeof(float));

        srccdf[0] = srcpdf[0];
        for (int i = 1; i < nensrc; i++) {
            srccdf[i] = srccdf[i - 1] + srcpdf[i];
        }

        float fnorm = 1.0f / srccdf[nensrc - 1];
        int binsok = 0;
        deltak = INVDIM; /* number of elements in inverse CDF */
        float gridsz = 1.0f / deltak;

        for (int i = 0; i < nensrc; i++) {
            srccdf[i] *= fnorm;
            if (i == 0) {
                if (srccdf[0] <= 3.0f * gridsz) {
                    binsok = 1;
                }
            }
            else {
                if ((srccdf[i] - srccdf[i - 1]) < 3.0f * gridsz) {
                    binsok = 1;
                }
            }
        }

        if (binsok != 0) {
            printf("Warning!, some of normalized bin probabilities are "
                "so small that bins may be missed.\n");
        }

        /* Calculate cdfinv. This array allows the rapid sampling for the
        energy by precomputing the results for a fine grid. */
        cdfinv1 = (float*)malloc(size_t(deltak * sizeof(float)));
        cdfinv2 = (float*)malloc(size_t(deltak * sizeof(float)));
        float ak;

        for (int k = 0; k < deltak; k++) {
            ak = (float)k * gridsz;
            int i;

            for (i = 0; i < nensrc; i++) {
                if (ak <= srccdf[i]) {
                    break;
                }
            }

            /* We should fall here only through the above break sentence. */
            if (i != 0) {
                cdfinv1[k] = ensrcd[i - 1];
            }
            else {
                cdfinv1[k] = enmin;
            }
            cdfinv2[k] = ensrcd[i] - cdfinv1[k];
        }

        /* Cleaning */
        fclose(fp);
        free(ensrcd);
        free(srcpdf);
        free(srccdf);

        break;
    }
}

Source::~Source() {
	free(cdfinv1);
	free(cdfinv2);
}

void Source::Initialize() {
    int ndeltak = (int)deltak;
	cudaMemcpyToSymbol(Spectrum::d_energy, &energy, sizeof(float));
	cudaMemcpyToSymbol(Spectrum::d_srctype, &source_type, sizeof(int));
	cudaMemcpyToSymbol(Spectrum::d_deltak, &ndeltak, sizeof(int));
	cudaMemcpyToSymbol(Spectrum::d_cdfinv1, cdfinv1, sizeof(float) * ndeltak);
	cudaMemcpyToSymbol(Spectrum::d_cdfinv2, cdfinv2, sizeof(float) * ndeltak);
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