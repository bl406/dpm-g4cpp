#----------------------------------------------------------------------------
# Setup the project
cmake_minimum_required(VERSION 3.18)
project (dpm++)

find_package(OpenMP)

enable_language(CUDA)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Xcompiler "/wd4819 /wd4996")

#----------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# Set the CXX flags and default build type
#
if (NOT EXISTS ${CMAKE_BINARY_DIR}/CMakeCache.txt)
  if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
  endif()
endif()

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W3 ${OpenMP_CXX_FLAGS}")

file ( GLOB headers_SIM
    ${CMAKE_SOURCE_DIR}*.hh
    ${CMAKE_SOURCE_DIR}/Simulation/inc/*.hh
  )
file ( GLOB cxx_sources_SIM
    ${CMAKE_SOURCE_DIR}/Simulation/src/*.cc
    ${CMAKE_SOURCE_DIR}/Simulation/src/*.cu
    ${CMAKE_SOURCE_DIR}/Simulation/src/SourceModels/*.cpp    
    ${CMAKE_SOURCE_DIR}/Simulation/src/SourceModels/*.cu
  )


#----------------------------------------------------------------------------
# Locate headers
#
include_directories ( ${CMAKE_SOURCE_DIR}/getopt                     
                      ${CMAKE_SOURCE_DIR}                     
                      ${CMAKE_SOURCE_DIR}/Utils/inc
                      ${CMAKE_SOURCE_DIR}/Simulation/inc/
                      ${CMAKE_SOURCE_DIR}/Simulation/src/SourceModels/
                      ${CUDAToolkit_INCLUDE_DIRS}
                    )


#----------------------------------------------------------------------------
# Build target
#
link_directories(${CMAKE_BINARY_DIR})

add_library(getopt SHARED ${CMAKE_SOURCE_DIR}/getopt/getopt.c)
target_compile_definitions(getopt PRIVATE EXPORTS_GETOPT)

add_executable ( dpm_Simulate ${CMAKE_SOURCE_DIR}/dpm_Simulate.cu ${headers_SIM} ${cxx_sources_SIM})
target_compile_options(dpm_Simulate PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(dpm_Simulate PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()
target_link_libraries (dpm_Simulate getopt.lib cudart_static.lib)
target_compile_definitions(dpm_Simulate PRIVATE _CRT_SECURE_NO_WARNINGS)
set_target_properties(dpm_Simulate PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(AllFiles ${cxx_sources_SIM} ${headers_SIM})
foreach(fileItem ${AllFiles})
    get_filename_component(PARENT_DIR "${fileItem}" DIRECTORY)
    string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}" "" GROUP "${PARENT_DIR}")
    string(REPLACE "/" "\\" GROUP "${GROUP}")
    set(GROUP "${GROUP}")
    source_group("${GROUP}" FILES "${fileItem}")
endforeach()

