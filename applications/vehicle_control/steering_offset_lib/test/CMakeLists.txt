project(steering_offset_sim)

add_executable(sim_main sim_main.cpp)
target_link_libraries(sim_main PRIVATE steering_offset_lib)
target_compile_features(sim_main PRIVATE cxx_std_17)
