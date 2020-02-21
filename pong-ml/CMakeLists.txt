
project(pong-ml LANGUAGES CXX)

include(CheckLanguage)
check_language(CUDA)
if (NOT CMAKE_CUDA_COMPILER)
    message("CUDA Support: No")
    message("    Using LibTorch with CPU-based Engine")
    set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/external/libtorch-linux")

    message("    To enable CUDA, install CUDA Toolkit 10.2 and define the following environment variables:")
    message("      * export CUDACXX=/usr/local/cuda/bin/nvcc")
    message("      * export CUDA_HOME=/usr/local/cuda")
    message("      * export CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda")
    message("      * cmake (...) -DLIBNVTOOLSEXT=/usr/local/cuda-10.2/targets/x86_64-linux/lib/libnvToolsExt.so")
else()
    message("CUDA Support: Yes")
    message("    Using LibTorch with CUDA-based Engine")
    set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/external/libtorch-cuda-linux")

    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 10)
    set(CMAKE_CUDA_STANDARD_REQUIRED True)

    if (NOT DEFINED CMAKE_CUDA_STANDARD)
        message(FATAL_ERROR "CUDA could not be enabled.")
        return()
    endif()

    if (NOT LIBNVTOOLSEXT)
        set(LIBNVTOOLSEXT /usr/local/cuda-10.2/targets/x86_64-linux/lib/libnvToolsExt.so)
        message(WARNING "Setting variable: LIBNVTOOLSEXT=${LIBNVTOOLSEXT}")
    endif()
endif()

find_package(Torch REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(SDL2 REQUIRED)
find_package(SDL2_image REQUIRED)
find_package(SDL2_ttf REQUIRED)

include_directories(
	${OPENGL_INCLUDE_DIRS}
	${GLEW_INCLUDE_DIRS}
	${SDL2_INCLUDE_DIRS}
	${SDL2_IMAGE_INCLUDE_DIRS}
	${SDL2_TTF_INCLUDE_DIRS}
	../external/glm/include
    ../external/Chipmunk-7.0.3/include)

add_executable(pong-ml
    basic_types.h
    board_renderer.cpp
    board_renderer.h
    board_sim.cpp
    board_sim.h
    config.h
    dq_bot.cpp
    dq_bot.h
    game_app.cpp
    game_app.h
    main.cpp
    pch.cpp
    pch.h)

target_link_libraries(pong-ml
    chipmunk_static
    ${TORCH_LIBRARIES}
    ${OPENGL_LIBRARIES}
    GLEW::GLEW
	${SDL2_LIBRARIES})
	#${SDL2_IMAGE_LIBRARIES}
	#${SDL2_TTF_LIBRARIES})

target_precompile_headers(pong-ml
    PUBLIC pch.h)

set_property(TARGET pong-ml PROPERTY CXX_STANDARD 17)
