cmake_minimum_required(VERSION 3.14)
project(TurboSMP LANGUAGES CXX CUDA)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

find_package(CUDAToolkit REQUIRED)
find_package(Boost COMPONENTS unit_test_framework program_options REQUIRED)
find_package(HDF5 COMPONENTS CXX REQUIRED)

file(GLOB SOURCE "src/*.c*")
add_executable(tsmp ${SOURCE})
target_include_directories(tsmp PUBLIC include)
target_link_libraries(tsmp
	PUBLIC
	CUDA::cudart CUDA::curand
	Boost::disable_autolinking Boost::dynamic_linking Boost::program_options
	${HDF5_LIBRARIES}
)

enable_testing()
add_subdirectory(test)
