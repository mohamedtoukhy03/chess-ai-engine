#include "uci.h"
#include "movegen.h"
#include "bitboard.h"
#include "perft.h"

#include <iostream>
#include <sstream>
#include <string>
#include <algorithm>

// ============================================================
// UCI Implementation
// ============================================================

UCI::UCI()
    : search_(evaluator_)
{
    pos_.set_startpos();
}

void UCI::loop() {
    std::string line;

    while (std::getline(std::cin, line)) {
        // Trim whitespace
        while (!line.empty() && line.back() == '\r')
            line.pop_back();

        if (line.empty()) continue;

        std::istringstream ss(line);
        std::string cmd;
        ss >> cmd;

        if (cmd == "uci")             cmd_uci();
        else if (cmd == "isready")    cmd_isready();
        else if (cmd == "ucinewgame") cmd_ucinewgame();
        else if (cmd == "position")   cmd_position(line.substr(cmd.size()));
        else if (cmd == "go")         cmd_go(line.substr(cmd.size()));
        else if (cmd == "setoption")  cmd_setoption(line.substr(cmd.size()));
        else if (cmd == "stop")       cmd_stop();
        else if (cmd == "quit")       { cmd_quit(); return; }
        else if (cmd == "d")          pos_.print();  // Debug: display board
        else if (cmd == "perft") {
            int depth = 5;
            ss >> depth;
            divide(pos_, depth);
        }
    }
}

void UCI::cmd_uci() {
    std::cout << "id name ChessAI 1.0" << std::endl;
    std::cout << "id author ChessEngine" << std::endl;
    std::cout << std::endl;
    std::cout << "option name ModelPath type string default models/chess_eval.onnx" << std::endl;
    std::cout << "option name Iterations type spin default 1000 min 100 max 100000" << std::endl;
    std::cout << "uciok" << std::endl;
}

void UCI::cmd_isready() {
    std::cout << "readyok" << std::endl;
}

void UCI::cmd_ucinewgame() {
    pos_.set_startpos();
}

void UCI::cmd_position(const std::string& args) {
    std::istringstream ss(args);
    std::string token;
    ss >> token;

    // "position startpos" or "position fen <fen>"
    if (token == "startpos") {
        pos_.set_startpos();
    } else if (token == "fen") {
        // Read the full FEN (up to 6 space-separated tokens)
        std::string fen;
        for (int i = 0; i < 6 && ss >> token; ++i) {
            if (token == "moves") {
                // Put "moves" back — handle below
                break;
            }
            if (!fen.empty()) fen += ' ';
            fen += token;
        }
        if (!fen.empty()) {
            pos_.set_fen(fen);
        }
        // token might be "moves" now
        if (token != "moves") {
            ss >> token; // try to read "moves"
        }
    }

    // Apply moves if present: "... moves e2e4 e7e5 ..."
    // Find "moves" in the rest of the string
    std::string remaining = ss.str();
    auto moves_pos = remaining.find("moves");
    if (moves_pos != std::string::npos) {
        std::istringstream move_ss(remaining.substr(moves_pos + 5));
        std::string move_str;
        while (move_ss >> move_str) {
            Move m = parse_move(move_str);
            if (m != MOVE_NONE) {
                UndoInfo undo;
                pos_.make_move(m, undo);
            }
        }
    }
}

void UCI::cmd_go(const std::string& args) {
    int time_ms = parse_time(args);

    // Default: 2 seconds if no time control specified
    if (time_ms <= 0) time_ms = 2000;

    Move best = search_.search(pos_, time_ms);

    if (best != MOVE_NONE) {
        std::cout << "bestmove " << best.to_string() << std::endl;
    } else {
        // No legal moves — shouldn't happen in normal play
        std::cout << "bestmove 0000" << std::endl;
    }
}

void UCI::cmd_setoption(const std::string& args) {
    std::istringstream ss(args);
    std::string token;

    std::string name, value;

    while (ss >> token) {
        if (token == "name") {
            name.clear();
            while (ss >> token && token != "value") {
                if (!name.empty()) name += ' ';
                name += token;
            }
        }
        if (token == "value") {
            value.clear();
            while (ss >> token) {
                if (!value.empty()) value += ' ';
                value += token;
            }
        }
    }

    if (name == "ModelPath") {
        evaluator_.load_model(value);
    }
}

void UCI::cmd_stop() {
    // In a threaded implementation, this would signal the search to stop
    // For now, search is synchronous
}

void UCI::cmd_quit() {
    // Cleanup if needed
}

// ============================================================
// Helpers
// ============================================================

Move UCI::parse_move(const std::string& str) const {
    if (str.size() < 4) return MOVE_NONE;

    Square from = string_to_square(str.substr(0, 2));
    Square to   = string_to_square(str.substr(2, 2));

    if (from == NO_SQUARE || to == NO_SQUARE) return MOVE_NONE;

    // Check for promotion
    if (str.size() >= 5) {
        PieceType promo = NO_PIECE_TYPE;
        switch (str[4]) {
            case 'q': promo = QUEEN;  break;
            case 'r': promo = ROOK;   break;
            case 'b': promo = BISHOP; break;
            case 'n': promo = KNIGHT; break;
        }
        if (promo != NO_PIECE_TYPE) {
            return Move::make_promotion(from, to, promo);
        }
    }

    // Check if this is a castling move
    Piece moving = pos_.piece_on(from);
    if (moving != NO_PIECE && piece_type(moving) == KING) {
        int diff = int(to) - int(from);
        if (diff == 2 || diff == -2) {
            return Move::make_castling(from, to);
        }
    }

    // Check if this is an en passant move
    if (moving != NO_PIECE && piece_type(moving) == PAWN) {
        if (to == pos_.ep_square() && square_file(from) != square_file(to)) {
            return Move::make_en_passant(from, to);
        }
    }

    return Move::make(from, to);
}

int UCI::parse_time(const std::string& args) const {
    std::istringstream ss(args);
    std::string token;

    int wtime = 0, btime = 0, winc = 0, binc = 0, movetime = 0;
    int movestogo = 30;

    while (ss >> token) {
        if (token == "wtime")     ss >> wtime;
        else if (token == "btime")     ss >> btime;
        else if (token == "winc")      ss >> winc;
        else if (token == "binc")      ss >> binc;
        else if (token == "movestogo") ss >> movestogo;
        else if (token == "movetime")  ss >> movetime;
        else if (token == "infinite")  return 60000;  // 1 minute
    }

    if (movetime > 0) return movetime;

    // Time management: allocate a portion of remaining time
    int our_time = (pos_.side_to_move() == WHITE) ? wtime : btime;
    int our_inc  = (pos_.side_to_move() == WHITE) ? winc  : binc;

    if (our_time > 0) {
        // Use 1/movestogo of remaining time + increment
        int allocated = our_time / movestogo + our_inc;
        // Don't use more than 50% of remaining time
        allocated = std::min(allocated, our_time / 2);
        // Minimum 100ms
        allocated = std::max(allocated, 100);
        return allocated;
    }

    return 0;
}
