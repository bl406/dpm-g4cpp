#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

template <class T>
void writeMetaImage(const std::string& baseFileName, const std::array<int, 3>& size, const std::array<float, 3>& spacing, T* imageData) {
    // �Զ������ļ���
    std::string mhdFile = baseFileName + ".mhd";
    std::string rawFile = baseFileName + ".bin";

    // �� .mhd �ļ�
    std::ofstream mhd(mhdFile);
    if (!mhd) {
        std::cerr << "Unable to open MHD file for writing." << std::endl;
        return;
    }

    // д�� .mhd �ļ�Ԫ����
    mhd << "ObjectType = Image" << std::endl;
    mhd << "NDims = 3" << std::endl;
    mhd << "DimSize = " << size[0] << " " << size[1] << " " << size[2] << std::endl;
    // �ж���������
    if (std::is_same<T, unsigned char>::value)
        mhd << "ElementType = MET_UCHAR" << std::endl;  // �������ͣ��޷����ַ�
    else if (std::is_same<T, short>::value)
        mhd << "ElementType = MET_SHORT" << std::endl;  // �������ͣ�������
    else if (std::is_same<T, float>::value)
        mhd << "ElementType = MET_FLOAT" << std::endl;  // �������ͣ���������
    mhd << "ElementDataFile = " << rawFile << std::endl;  // ָ��ʵ�������ļ�������
    mhd << "ElementSpacing = " << spacing[0] << " " << spacing[1] << " " << spacing[2] << std::endl;  // �������ؼ��Ϊ1.0����λ���Ե�����

    mhd.close();
    //std::cout << ".mhd file written successfully: " << mhdFile << std::endl;

    // �� .raw �ļ�
    std::ofstream raw(rawFile, std::ios::binary);
    if (!raw) {
        std::cerr << "Unable to open RAW file for writing." << std::endl;
        return;
    }

    // ֱ��һ����д��ͼ�����ݵ� .raw �ļ�
    raw.write(reinterpret_cast<const char*>(imageData), size[0]* size[1]* size[2] * sizeof(T));

    raw.close();
    //std::cout << ".raw file written successfully: " << rawFile << std::endl;
}