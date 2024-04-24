project(xcorrFFT_example)

add_executable(simple simple.cpp)
target_link_libraries(simple PUBLIC ${ippcorelib} ${ippslib} ${ippvmlib})
