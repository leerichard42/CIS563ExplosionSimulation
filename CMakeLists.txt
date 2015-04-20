cmake_minimum_required (VERSION 3.0.0)

# on OSX we have to explicitly set clang/clang++
set (CMAKE_C_COMPILER clang)
set (CMAKE_CXX_COMPILER clang++)

project (SmokeSimulation)
list (APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/external")

set (EXTERNAL "external")

# Set up include and lib paths
include_directories (${EXTERNAL}/include)

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	set (CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} ${EXTERNAL}/lib/osx)
endif()

set (GLFW_INCLUDE_DIR ${EXTERNAL}/include)
set (GLFW_LIBRARY_DIR ${CMAKE_LIBRARY_PATH})
set (GLEW_INCLUDE_DIR ${EXTERNAL}/include)
set (GLEW_LIBRARY_DIR ${CMAKE_LIBRARY_PATH})

# Find and set up core dependency libs
find_library (GLFW_LIBRARY "glfw3" HINTS ${GLFW_LIBRARY_DIR})
find_package (OPENGL)
find_package (GLEW)

if(${CMAKE_SYSTEM_NAME} MATCHES "Linux" OR ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  find_package(GLUT)
endif()

set (CORELIBS ${GLFW_LIBRARY} ${GLUT_LIBRARY} ${OPENGL_LIBRARY} ${GLEW_LIBRARY})

#OSX-specific hacks/fixes
if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
   	#Link IOKit because this is where we get GL stuff for OSX
   	set (IOKIT "-framework IOKit")
	set (CORELIBS ${CORELIBS} ${IOKIT})
endif (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

#Compiler flag magic
if (${CMAKE_SYSTEM_NAME} MATCHES "Linux" OR ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -m64 -msse2")
elseif (WIN32)
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif ()

# Add all source files. Headers don't need to be listed here since the compiler will find them;
# we just need the actual files being fed directly to the compiler
set (SOURCE_FILES "src/main.cpp"
                  "src/basic_math.cpp"
                  "src/camera.cpp"
                  "src/constants.cpp"
                  "src/fps.cpp"
                  "src/grid_data.cpp"
                  "src/mac_grid.cpp"
                  "src/smoke_sim.cpp"
                  "src/stb_image.c"
                  "src/stb_image_write.c")

add_executable(smokeSimulation ${SOURCE_FILES})

target_link_libraries (smokeSimulation ${CORELIBS})
