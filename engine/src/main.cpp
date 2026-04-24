#include "bitboard.h"
#include "position.h"
#include "movegen.h"
#include "perft.h"
#include "uci.h"

#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>
#include <vector>
#include <cstring>

// ============================================================
// Perft test suite
// ============================================================

struct PerftTest {
    std::string fen;
    std::string name;
    std::vector<std::pair<int, uint64_t>> expected;
};

static bool run_test(const PerftTest& test) {
    Position pos;
    pos.set_fen(test.fen);

    std::cout << "=== " << test.name << " ===\n";
    std::cout << "FEN: " << test.fen << "\n\n";

    bool all_pass = true;

    for (auto& [depth, expected] : test.expected) {
        auto t0 = std::chrono::high_resolution_clock::now();
        uint64_t result = perft(pos, depth);
        auto t1 = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double mnps = (ms > 0) ? (result / ms / 1000.0) : 0;

        bool pass = (result == expected);
        all_pass &= pass;

        std::cout << "  Depth " << depth << ": "
                  << std::setw(12) << result
                  << (pass ? "  PASS" : "  FAIL")
                  << "  (" << std::fixed << std::setprecision(1)
                  << ms << " ms, "
                  << std::setprecision(2) << mnps << " Mnps)";
        if (!pass)
            std::cout << "  [expected " << expected << "]";
        std::cout << "\n";
    }

    std::cout << "\n";
    return all_pass;
}

static int run_perft_suite() {
    std::cout << "Chess Engine - Perft Validation Suite\n";
    std::cout << "=====================================\n\n";

    std::vector<PerftTest> tests = {
        {
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "Initial Position",
            {{1, 20}, {2, 400}, {3, 8902}, {4, 197281}, {5, 4865609}, {6, 119060324}}
        },
        {
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "Kiwipete",
            {{1, 48}, {2, 2039}, {3, 97862}, {4, 4085603}, {5, 193690690}}
        },
        {
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "Position 3 (Endgame)",
            {{1, 14}, {2, 191}, {3, 2812}, {4, 43238}, {5, 674624}}
        },
        {
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            "Position 4 (Promotions)",
            {{1, 6}, {2, 264}, {3, 9467}, {4, 422333}, {5, 15833292}}
        },
        {
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
            "Position 5 (Discovered Check)",
            {{1, 44}, {2, 1486}, {3, 62379}, {4, 2103487}, {5, 89941194}}
        },
    };

    int passed = 0, failed = 0;
    for (auto& test : tests) {
        if (run_test(test))
            passed++;
        else
            failed++;
    }

    std::cout << "=====================================\n";
    std::cout << "Results: " << passed << " passed, "
              << failed << " failed out of "
              << tests.size() << " test positions\n\n";

    if (failed == 0)
        std::cout << "ALL PERFT TESTS PASSED!\n";
    else
        std::cout << "SOME TESTS FAILED - move generation has bugs.\n";

    return failed > 0 ? 1 : 0;
}

// ============================================================
// Main: UCI mode (default) or perft mode (--perft flag)
// ============================================================

int main(int argc, char* argv[]) {
    BB::init();

    // Check for --perft flag
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--perft") == 0) {
            return run_perft_suite();
        }
    }

    // Default: UCI mode
    UCI uci;
    uci.loop();

    return 0;
}
