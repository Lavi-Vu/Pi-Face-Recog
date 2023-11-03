cmake_minimum_required(VERSION 3.9)

project(FaceRecognition)
include(FetchContent)
FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz)
FetchContent_MakeAvailable(json)

macro(use_cxx11)
  if (CMAKE_VERSION VERSION_LESS "3.1")
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
    endif ()
  else ()
    set (CMAKE_CXX_STANDARD 17)
  endif ()
endmacro(use_cxx11)

find_package(OpenCV REQUIRED)                              
find_package(CURL REQUIRED)

set(SOURCES src/main.cpp src/TMtCNN.cpp src/TBlur.cpp src/TLive.cpp src/TRetina.cpp src/TWarp.cpp src/TArcface.cpp)

set(EXTRA_LIBS ${OpenCV_LIBS} /usr/local/lib/ncnn/libncnn.a)

set(EXECUTABLE_OUTPUT_PATH "./")

include_directories(include ${OpenCV_INCLUDE_DIRS} /usr/local/include/ncnn)
add_link_options(-fopenmp -s -ldl -lpthread)

add_executable(FaceRecognition ${SOURCES})
# add_executable(testMain src/testMain.cpp ${SOURCES})

target_link_libraries(FaceRecognition ${EXTRA_LIBS} CURL::libcurl nlohmann_json::nlohmann_json sioclient) 
# target_link_libraries(testMain ${EXTRA_LIBS} CURL::libcurl nlohmann_json::nlohmann_json sioclient) 
