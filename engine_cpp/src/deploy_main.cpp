#include "engine_api.h"

#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    std::string fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    std::string model = "models/chess_eval.onnx";
    int depth = 4;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--fen" && i + 1 < argc) {
            fen = argv[++i];
        } else if (arg == "--depth" && i + 1 < argc) {
            depth = std::stoi(argv[++i]);
        } else if (arg == "--model" && i + 1 < argc) {
            model = argv[++i];
        }
    }

    std::cout << get_best_move(fen, depth, model) << std::endl;
    return 0;
}
