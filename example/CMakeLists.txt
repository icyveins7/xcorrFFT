project(xcorrFFT_example)

if (${IMPL} STREQUAL IPP)
    add_executable(simple_ipp simple_ipp.cpp)
    target_link_libraries(simple_ipp PUBLIC ${ippcorelib} ${ippslib} ${ippvmlib})
endif()
