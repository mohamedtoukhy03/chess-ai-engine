#pragma once

#include "position.h"
#include "search.h"
#include "nn_eval.h"

#include <string>

// ============================================================
// Universal Chess Interface (UCI) Protocol Handler
// ============================================================

class UCI {
public:
    UCI();

    // Main UCI loop — reads from stdin, writes to stdout
    void loop();

private:
    Position pos_;
    NNEvaluator evaluator_;
    AlphaBetaSearch search_;

    // UCI command handlers
    void cmd_uci();
    void cmd_isready();
    void cmd_ucinewgame();
    void cmd_position(const std::string& args);
    void cmd_go(const std::string& args);
    void cmd_setoption(const std::string& args);
    void cmd_stop();
    void cmd_quit();

    // Helpers
    Move parse_move(const std::string& str) const;
    int parse_time(const std::string& args) const;
};
