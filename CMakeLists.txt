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
    message("Adding ipp_ext submodule")
    include_directories(
        ipp_ext/include
    )
    add_compile_definitions(IPP_IMPL)

    # Find all IPP related includes and libs
    # Do this to include all the IPP paths for headers, assumed to be done by the normal vars.bat provided by oneAPI
    include_directories($ENV{INCLUDE} $ENV{CPATH} ${CMAKE_CURRENT_SOURCE_DIR}/../include/)
    # On linux it's CPATH, on win it's INCLUDE

    # Define the library paths for IPP
    if (WIN32)
        message("Finding IPP libraries for Windows...")
        find_library(ippcorelib NAMES ippcore ENV LIBRARY_PATH)
        find_library(ippslib NAMES ipps ENV LIBRARY_PATH) # Did i actually need HINTS here too?
        find_library(ippvmlib NAMES ippvm ENV LIBRARY_PATH) # Did i actually need HINTS here too?
        message("Libpath: " ${ippcorelib})
        message("Libpath: " ${ippslib})
        message("Libpath: " ${ippvmlib})
    elseif (APPLE)
        message("Finding IPP libraries for MacOS...")
        find_library(ippcorelib NAMES libippcore.a HINTS ENV DYLD_LIBRARY_PATH) # on mac it seems like it's here, even though it's linux based
        find_library(ippslib NAMES libipps.a HINTS ENV DYLD_LIBRARY_PATH)
        find_library(ippvmlib NAMES libippvm.a HINTS ENV DYLD_LIBRARY_PATH)
        message("Libpath: " ${ippcorelib})
        message("Libpath: " ${ippslib})
        message("Libpath: " ${ippvmlib})
    else ()
        message("Finding IPP libraries for Linux...")
        find_library(ippcorelib NAMES libippcore ippcore HINTS ENV LD_LIBRARY_PATH)
        find_library(ippslib NAMES libipps ipps HINTS ENV LD_LIBRARY_PATH)
        find_library(ippvmlib NAMES libippvm ippvm HINTS ENV LD_LIBRARY_PATH)
        message("Libpath: " ${ippcorelib})
        message("Libpath: " ${ippslib})
        message("Libpath: " ${ippvmlib})
    endif()
else()
    message(FATAL_ERROR "Other implementations not yet supported.")
endif()

# Add subdirectories
add_subdirectory(example)
