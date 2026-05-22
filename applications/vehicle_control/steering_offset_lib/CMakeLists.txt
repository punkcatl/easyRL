cmake_minimum_required(VERSION 3.14)
project(steering_offset_lib)

file(GLOB_RECURSE CPP_FILES src/*.cpp)

add_library(${PROJECT_NAME} STATIC ${CPP_FILES})
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)

add_subdirectory(test)
