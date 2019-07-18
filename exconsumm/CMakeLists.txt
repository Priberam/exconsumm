# CMakeList.txt : CMake project for exconsumm, include source and define
# project specific logic here.
#

set(CMAKE_CXX_FLAGS "-Wall -std=c++11 -O3 -g")

project(exconsumm)
cmake_minimum_required (VERSION 3.8)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -DLIBCONFIGXX_EXPORTS -DLIBCONFIG_STATIC")
# look for Eigen
if (MKL OR MKL_ROOT)
  if (DEFINED ENV{MKL_ROOT} AND NOT DEFINED MKL_ROOT)  # use env variable if not defined
    set(MKL_ROOT $ENV{MKL_ROOT})
  endif()
  find_mkl()  # sets include/lib directories and sets ${LIBS} needed for linking
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEIGEN_USE_MKL_ALL")
endif()
find_package(Eigen3 REQUIRED)
link_directories(${MKL_LINK_DIRS})

# Add source to this project's executable.
add_executable (exconsumm "NeuralSummarization.cpp" dynet_binary_io.cpp
OSListdir.cpp
Corpus.cpp
FastRouge.cpp
RefreshCNNDMCorpus.cpp
NoNameYetModel.cpp
)
if (UNIX)
    target_link_libraries(exconsumm stdc++fs)
endif (UNIX)
target_include_directories(exconsumm PRIVATE .. ../dynet ${EIGEN3_INCLUDE_DIR} ../libconfig)
target_link_libraries(exconsumm dynet config++)


