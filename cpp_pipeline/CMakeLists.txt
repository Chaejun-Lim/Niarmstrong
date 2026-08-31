cmake_minimum_required(VERSION 3.16)
project(hybrid_pipeline LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-O3 -Wall -Wextra -Wpedantic)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(/usr/local/zed/include)
include_directories(/usr/local/cuda/include)

add_executable(control_service src/control_service.cpp)
add_executable(camera_service_cpp src/camera_service_stub.cpp)
add_executable(shm_monitor src/shm_monitor.cpp)

target_link_libraries(control_service rt pthread)
target_link_directories(camera_service_cpp PRIVATE /usr/local/zed/lib)
target_link_libraries(camera_service_cpp sl_zed rt pthread)
set_target_properties(camera_service_cpp PROPERTIES BUILD_RPATH "/usr/local/zed/lib")
target_link_libraries(shm_monitor rt pthread)
