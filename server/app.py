"""
Chess AI Backend — Python/FastAPI

Manages the C++ UCI engine as a subprocess and exposes a REST API
for the web frontend. Also serves the frontend static files.

Endpoints:
    POST /api/move      — Get the engine's best move
    POST /api/newgame   — Start a new game
    GET  /api/status    — Check engine status
    GET  /               — Serve the frontend
"""

import os
import asyncio
import subprocess
import logging
import time
import queue
import threading
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ============================================================
# Configuration
# ============================================================

ENGINE_PATH = os.environ.get(
    "ENGINE_PATH",
    str(Path(__file__).parent.parent / "build" / "engine" / "chess_engine")
)
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    str(Path(__file__).parent.parent / "models" / "chess_eval.onnx")
)
FRONTEND_DIR = str(Path(__file__).parent.parent / "frontend")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chess-server")

# ============================================================
# UCI Engine Manager
# ============================================================

class UCIEngine:
    """Manages the C++ chess engine subprocess via UCI protocol."""

    def __init__(self):
        self.process: subprocess.Popen | None = None
        self.initialized = False
        self.lock = threading.Lock()
        self.stdout_queue: queue.Queue[str] = queue.Queue()
        self.stdout_thread: threading.Thread | None = None

    def start(self):
        """Start the engine subprocess."""
        if not Path(ENGINE_PATH).exists():
            logger.warning(f"Engine not found at {ENGINE_PATH}")
            logger.warning("Build the engine first: cd build && cmake .. && make")
            return False

        try:
            self.process = subprocess.Popen(
                [ENGINE_PATH],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )

            self.stdout_queue = queue.Queue()
            self.stdout_thread = threading.Thread(target=self._stdout_reader, daemon=True)
            self.stdout_thread.start()

            # Initialize UCI handshake
            self._send("uci")
            if self._wait_for("uciok", timeout=5.0):
                # Load NN model if available
                if Path(MODEL_PATH).exists():
                    self._send(f"setoption name ModelPath value {MODEL_PATH}")

                self._send("isready")
                if self._wait_for("readyok", timeout=5.0):
                    self.initialized = True
                    logger.info("Engine initialized successfully")
                    return True

            logger.error("Engine failed UCI handshake")
            return False

        except Exception as e:
            logger.error(f"Failed to start engine: {e}")
            return False

    def stop(self):
        """Stop the engine subprocess."""
        if self.process:
            try:
                self._send("quit")
                self.process.wait(timeout=3)
            except Exception:
                self.process.kill()
            self.process = None
            self.initialized = False
            self.stdout_thread = None

    def get_best_move(self, fen: str | None, moves: str | None, time_ms: int) -> str | None:
        """
        Send a position to the engine and get the best move.

        Args:
            fen: FEN string (None or "startpos" for starting position)
            moves: Space-separated UCI moves from startpos
            time_ms: Time to think in milliseconds

        Returns:
            Best move in UCI format (e.g., "e2e4") or None on failure
        """
        with self.lock:
            if not self.initialized:
                return None

            # Build position command
            if fen is None or fen == "startpos":
                pos_cmd = "position startpos"
            else:
                pos_cmd = f"position fen {fen}"

            if moves:
                pos_cmd += f" moves {moves}"

            self._send(pos_cmd)
            self._send(f"go movetime {time_ms}")

            # Read until "bestmove"
            response = self._wait_for("bestmove", timeout=(time_ms / 1000.0) + 5.0)
            if response and response.startswith("bestmove"):
                parts = response.split()
                return parts[1] if len(parts) > 1 else None

            return None
    def new_game(self):
        """Signal a new game to the engine."""
        with self.lock:
            if self.initialized:
                self._send("ucinewgame")
                self._send("isready")
                self._wait_for("readyok", timeout=5.0)

    def _send(self, cmd: str):
        """Send a command to the engine's stdin."""
        if self.process and self.process.stdin:
            try:
                self.process.stdin.write(cmd + "\n")
                self.process.stdin.flush()
            except Exception as e:
                logger.error(f"Failed to send '{cmd}': {e}")

    def _stdout_reader(self):
        """Continuously read engine stdout and enqueue lines."""
        if not self.process or not self.process.stdout:
            return

        try:
            while self.process and self.process.stdout:
                line = self.process.stdout.readline()
                if not line:
                    if self.process.poll() is not None:
                        break
                    continue
                self.stdout_queue.put(line.strip())
        except Exception:
            return

    def _wait_for(self, keyword: str, timeout: float = 5.0) -> str | None:
        """Read lines from engine stdout until one contains the keyword."""
        if not self.process:
            return None

        end_time = time.monotonic() + timeout
        while time.monotonic() < end_time:
            remaining = max(0.0, end_time - time.monotonic())
            try:
                line = self.stdout_queue.get(timeout=remaining)
            except queue.Empty:
                continue

            if not line:
                continue

            logger.debug(f"Engine: {line}")
            if keyword in line:
                return line

        return None


# ============================================================
# FastAPI Application
# ============================================================

engine = UCIEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start engine on startup, stop on shutdown."""
    engine.start()
    yield
    engine.stop()


app = FastAPI(
    title="Chess AI",
    description="Neural network-powered chess engine API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Request / Response models
# ============================================================

class MoveRequest(BaseModel):
    fen: str | None = "startpos"
    moves: str | None = ""
    timeMs: int = 2000


class MoveResponse(BaseModel):
    bestMove: str
    success: bool
    error: str | None = None


class StatusResponse(BaseModel):
    initialized: bool
    engine: str = "ChessAI 1.0"


class MessageResponse(BaseModel):
    success: bool
    message: str


# ============================================================
# API Endpoints
# ============================================================

@app.post("/api/move", response_model=MoveResponse)
async def get_move(request: MoveRequest):
    """Get the engine's best move for a given position."""
    best_move = await asyncio.to_thread(
        engine.get_best_move, request.fen, request.moves, request.timeMs
    )

    if best_move:
        return MoveResponse(bestMove=best_move, success=True)
    else:
        return MoveResponse(
            bestMove="",
            success=False,
            error="Engine failed to return a move"
        )


@app.post("/api/newgame", response_model=MessageResponse)
async def new_game():
    """Start a new game."""
    await asyncio.to_thread(engine.new_game)
    return MessageResponse(success=True, message="New game started")


@app.get("/api/status", response_model=StatusResponse)
async def status():
    """Check engine status."""
    return StatusResponse(initialized=engine.initialized)


# ============================================================
# Serve frontend static files
# ============================================================

# Serve frontend at root
if Path(FRONTEND_DIR).exists():
    @app.get("/")
    async def serve_index():
        return FileResponse(Path(FRONTEND_DIR) / "index.html")

    app.mount("/", StaticFiles(directory=FRONTEND_DIR), name="frontend")
