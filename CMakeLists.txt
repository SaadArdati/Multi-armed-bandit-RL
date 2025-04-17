cmake_minimum_required(VERSION 3.30)
project(Multi_armed_bandit_RL)

set(CMAKE_CXX_STANDARD 20)

# OpenMP configuration for macOS
if(APPLE)
    if(CMAKE_C_COMPILER_ID MATCHES "Clang")
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp")
        set(OpenMP_C_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib")
    endif()
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib")
    endif()
    # Add include path for omp.h
    include_directories("/opt/homebrew/opt/libomp/include")
endif()


find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)

set(CMAKE_C_COMPILER ${MPI_C_COMPILER})
set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})


include_directories(include)

add_executable(Multi_armed_bandit_RL
    include/Bandit.h
    include/BernoulliBandit.h
    include/Experiment.h
    include/NormalBandit.h
    src/Bandit.cc
    src/BernoulliBandit.cc
    src/Experiment.cc
    src/NormalBandit.cc
    mainBernoulli.cc
    mainNormal.cc)


target_link_libraries(mainBernoulli PRIVATE OpenMP::OpenMP_CXX)
