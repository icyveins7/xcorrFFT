cmake_minimum_required(VERSION 3.20)

project(xcorrFFT)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(
    include/
)

# Select implementation
if (DEFINED IMPL)
    message("Implementation chosen to be " ${IMPL})
else()
    # Default to IPP
    set(IMPL IPP)
    message("Implementation defaulting to " ${IMPL})
endif()

# If IPP selected, add the ipp_ext submodule
if (${IMPL} STREQUAL IPP)
    message("Adding IPP includes at " $ENV{IPPROOT}/include)
    message("Adding ipp_ext submodule")
    include_directories(
        $ENV{IPPROOT}/include
        ipp_ext/include
    )
    add_compile_definitions(IPP_IMPL)
else()
    message(FATAL_ERROR "Other implementations not yet supported.")
endif()

# Add subdirectories
add_subdirectory(example)
