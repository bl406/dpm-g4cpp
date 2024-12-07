#----------------------------------------------------------------------------
# Setup the project
cmake_minimum_required(VERSION 3.18)
project (dpm++)


# 启用CUDA语言
enable_language(CUDA)
# 查找CUDA
find_package(CUDA REQUIRED)
# 设置CUDA编译选项
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
    -O3 # 优化
    -g   # 调试信息
    -arch=sm_70 # 针对的GPU架构，根据需要修改
  )


#----------------------------------------------------------------------------
# Find Geant4 package
#
find_package ( Geant4 REQUIRED )


#----------------------------------------------------------------------------
# Setup Geant4 include directories and compile definitions
#
include ( ${Geant4_USE_FILE} )


#-------------------------------------------------------------------------------
# Set the CXX flags and default build type
#
if (NOT EXISTS ${CMAKE_BINARY_DIR}/CMakeCache.txt)
  if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
  endif()
endif()
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W3")



#-------------------------------------------------------------------------------
# Set the headers and sources for the data GENeration an SIMulation parts
#
file ( GLOB headers_GEN
    ${CMAKE_SOURCE_DIR}/DataInit/inc/*.hh
    ${CMAKE_SOURCE_DIR}/Utils/inc/*.hh
  )
file ( GLOB cxx_sources_GEN
    ${CMAKE_SOURCE_DIR}/DataInit/src/*.cc
    ${CMAKE_SOURCE_DIR}/Utils/src/*.cc
  )

file ( GLOB headers_SIM
    ${CMAKE_SOURCE_DIR}*.hh
    ${CMAKE_SOURCE_DIR}/Simulation/inc/*.hh
  )
file ( GLOB cxx_sources_SIM
    ${CMAKE_SOURCE_DIR}/Simulation/src/*.cc
    ${CMAKE_SOURCE_DIR}/Simulation/src/*.cu
  )
file ( GLOB validation_sources
        ${CMAKE_SOURCE_DIR}/modelValidationTests/src/*.cc
        )


#----------------------------------------------------------------------------
# Locate headers
#
include_directories ( ${CMAKE_SOURCE_DIR}/getopt
                      ${Geant4_INCLUDE_DIR}
                      ${CMAKE_SOURCE_DIR}
                      ${CMAKE_SOURCE_DIR}/DataInit/inc
                      ${CMAKE_SOURCE_DIR}/Utils/inc
                      ${CMAKE_SOURCE_DIR}/Simulation/inc/
                      ${CMAKE_SOURCE_DIR}/modelValidationTests/inc/
                      ${CUDAToolkit_INCLUDE_DIRS}
                    )


#----------------------------------------------------------------------------
# Build target
#

link_directories(${CMAKE_BINARY_DIR})

add_library(getopt SHARED ${CMAKE_SOURCE_DIR}/getopt/getopt.c)
target_compile_definitions(getopt PRIVATE EXPORTS_GETOPT)

set(GETOPT_LIBRARIES getopt.lib)

add_executable ( dpm_GenerateData ${CMAKE_SOURCE_DIR}/dpm_GenerateData.cc ${cxx_sources_GEN})
target_link_libraries (dpm_GenerateData  ${Geant4_LIBRARIES})

cuda_add_executable ( dpm_Simulate ${CMAKE_SOURCE_DIR}/dpm_Simulate.cu ${cxx_sources_SIM})
# target_compile_options(dpm_Simulate PRIVATE ${CUDA_NVCC_FLAGS})
target_link_libraries (dpm_Simulate ${GETOPT_LIBRARIES} cudart_static.lib)
target_compile_definitions(dpm_Simulate PRIVATE _CRT_SECURE_NO_WARNINGS)
set_target_properties(dpm_Simulate PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)

## add model level test for bremsstrahlung
add_executable ( test_brem ${CMAKE_SOURCE_DIR}/test_brem.cc ${cxx_sources_SIM})
target_link_libraries (test_brem ${GETOPT_LIBRARIES})

add_library(validation STATIC ${validation_sources} ${cxx_sources_SIM})
target_link_libraries(validation ${GETOPT_LIBRARIES})

add_executable(modeltests model_validation.cpp)
target_link_libraries (modeltests validation ${GETOPT_LIBRARIES})

install (TARGETS validation DESTINATION ${CMAKE_SOURCE_DIR}/lib)
install (FILES  ${CMAKE_SOURCE_DIR}/modelValidationTests/inc/testAPI.h DESTINATION ${CMAKE_SOURCE_DIR}/lib)

