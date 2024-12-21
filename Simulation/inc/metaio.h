#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

template <class T>
void writeMetaImage(const std::string& baseFileName, const std::array<int, 3>& size, const std::array<float, 3>& spacing, T* imageData) {
    // 自动生成文件名
    std::string mhdFile = baseFileName + ".mhd";
    std::string rawFile = baseFileName + ".bin";

    // 打开 .mhd 文件
    std::ofstream mhd(mhdFile);
    if (!mhd) {
        std::cerr << "Unable to open MHD file for writing." << std::endl;
        return;
    }

    // 写入 .mhd 文件元数据
    mhd << "ObjectType = Image" << std::endl;
    mhd << "NDims = 3" << std::endl;
    mhd << "DimSize = " << size[0] << " " << size[1] << " " << size[2] << std::endl;
    // 判读数据类型
    if (std::is_same<T, unsigned char>::value)
        mhd << "ElementType = MET_UCHAR" << std::endl;  // 数据类型：无符号字符
    else if (std::is_same<T, short>::value)
        mhd << "ElementType = MET_SHORT" << std::endl;  // 数据类型：短整型
    else if (std::is_same<T, float>::value)
        mhd << "ElementType = MET_FLOAT" << std::endl;  // 数据类型：浮动数据
    mhd << "ElementDataFile = " << rawFile << std::endl;  // 指向实际数据文件的名称
    mhd << "ElementSpacing = " << spacing[0] << " " << spacing[1] << " " << spacing[2] << std::endl;  // 假设像素间距为1.0（单位可以调整）

    mhd.close();
    //std::cout << ".mhd file written successfully: " << mhdFile << std::endl;

    // 打开 .raw 文件
    std::ofstream raw(rawFile, std::ios::binary);
    if (!raw) {
        std::cerr << "Unable to open RAW file for writing." << std::endl;
        return;
    }

    // 直接一次性写入图像数据到 .raw 文件
    raw.write(reinterpret_cast<const char*>(imageData), size[0]* size[1]* size[2] * sizeof(T));

    raw.close();
    //std::cout << ".raw file written successfully: " << rawFile << std::endl;
}