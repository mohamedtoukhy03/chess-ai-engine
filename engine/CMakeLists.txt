# Chess core static library
add_library(chess_core STATIC
    src/bitboard.cpp
    src/position.cpp
    src/movegen.cpp
    src/nn_eval.cpp
    src/search.cpp
    src/uci.cpp
    src/perft.cpp
)
target_include_directories(chess_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Optional ONNX Runtime support
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
    PATHS /usr/include/onnxruntime /usr/local/include/onnxruntime
          /usr/include /usr/local/include
)
find_library(ONNXRUNTIME_LIB onnxruntime
    PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu
)

if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIB)
    message(STATUS "ONNX Runtime found: ${ONNXRUNTIME_LIB}")
    target_compile_definitions(chess_core PUBLIC USE_ONNXRUNTIME)
    target_include_directories(chess_core PUBLIC ${ONNXRUNTIME_INCLUDE_DIR})
    target_link_libraries(chess_core PUBLIC ${ONNXRUNTIME_LIB})
else()
    message(STATUS "ONNX Runtime not found — using material evaluation fallback")
endif()

# Main executable (UCI engine + perft)
add_executable(chess_engine src/main.cpp)
target_link_libraries(chess_engine PRIVATE chess_core)
