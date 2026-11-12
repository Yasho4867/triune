cmake_minimum_required(VERSION 3.18)
project(triune_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(triune_core SHARED
    engine/engine.cpp
)

target_include_directories(triune_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/engine)

find_package(OpenMP)

if(MSVC)
    target_compile_options(triune_core PRIVATE /O2 /Oi /Ot /arch:AVX2 /EHsc)
    if(OpenMP_CXX_FOUND)
        target_compile_options(triune_core PRIVATE ${OpenMP_CXX_FLAGS})
        target_link_libraries(triune_core PRIVATE OpenMP::OpenMP_CXX)
    endif()
else()
    target_compile_options(triune_core PRIVATE -O3 -mavx2 -mfma -fPIC)
    if(OpenMP_CXX_FOUND)
        target_compile_options(triune_core PRIVATE ${OpenMP_CXX_FLAGS})
        target_link_libraries(triune_core PRIVATE OpenMP::OpenMP_CXX)
    endif()
    target_link_libraries(triune_core PRIVATE dl)
endif()
