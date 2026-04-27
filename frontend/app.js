// ============================================================
// Chess AI — Frontend Application
// ============================================================

// When served by FastAPI, use relative path (same origin)
// When opening index.html directly, fall back to localhost:8000
const API_BASE = window.location.port ? '/api' : 'http://localhost:8000/api';

let board = null;
let game = new Chess();
let moveHistory = [];
let playerColor = 'white';
let thinkTime = 2000;
let gameInProgress = true;

// ============================================================
// Board initialization
// ============================================================

function initBoard() {
    const config = {
        draggable: true,
        position: 'start',
        onDragStart: onDragStart,
        onDrop: onDrop,
        onSnapEnd: onSnapEnd,
        onMouseoutSquare: onMouseoutSquare,
        onMouseoverSquare: onMouseoverSquare,
        pieceTheme: 'https://chessboardjs.com/img/chesspieces/wikipedia/{piece}.png',
        appearSpeed: 'fast',
        moveSpeed: 200,
        snapbackSpeed: 300,
    };

    board = Chessboard('chessBoard', config);
    $(window).on('resize', () => board.resize());
}

// ============================================================
// Highlighting & Move Indicators
// ============================================================

function removeGreySquares() {
    $('#chessBoard .square-55d63').removeClass('square-move-dest');
}

function greySquare(square) {
    $('#chessBoard .square-' + square).addClass('square-move-dest');
}

function onMouseoverSquare(square, piece) {
    if (game.game_over() || !gameInProgress) return;

    // Only allow highlighting the player's own pieces
    if (playerColor === 'white' && piece && piece.search(/^b/) !== -1) return;
    if (playerColor === 'black' && piece && piece.search(/^w/) !== -1) return;

    // Only allow moves when it's the player's turn
    if (playerColor === 'white' && game.turn() !== 'w') return;
    if (playerColor === 'black' && game.turn() !== 'b') return;

    var moves = game.moves({
        square: square,
        verbose: true
    });

    if (moves.length === 0) return;

    for (var i = 0; i < moves.length; i++) {
        greySquare(moves[i].to);
    }
}

function onMouseoutSquare(square, piece) {
    removeGreySquares();
}

function updateCheckHighlight() {
    $('#chessBoard .square-55d63').removeClass('square-in-check');

    if (game.in_check() || game.in_checkmate()) {
        const turn = game.turn();
        
        // Find king square manually since chess.js doesn't expose it directly
        const boardState = game.board();
        const files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
        
        for (let r = 0; r < 8; r++) {
            for (let c = 0; c < 8; c++) {
                const piece = boardState[r][c];
                if (piece && piece.type === 'k' && piece.color === turn) {
                    const square = files[c] + (8 - r);
                    $('#chessBoard .square-' + square).addClass('square-in-check');
                    return;
                }
            }
        }
    }
}

// ============================================================
// Drag & drop handlers
// ============================================================

function onDragStart(source, piece, position, orientation) {
    if (game.game_over()) return false;
    if (!gameInProgress) return false;

    if (playerColor === 'white' && piece.search(/^b/) !== -1) return false;
    if (playerColor === 'black' && piece.search(/^w/) !== -1) return false;

    if (playerColor === 'white' && game.turn() !== 'w') return false;
    if (playerColor === 'black' && game.turn() !== 'b') return false;

    return true;
}

function onDrop(source, target) {
    removeGreySquares();
    
    let move = game.move({
        from: source,
        to: target,
        promotion: 'q'
    });

    if (move === null) return 'snapback';

    addMoveToHistory(move);
    updateGameStatus();
    updateCheckHighlight();

    if (game.game_over()) {
        handleGameOver();
        return;
    }

    setTimeout(requestAIMove, 250);
}

function onSnapEnd() {
    board.position(game.fen());
}

// ============================================================
// AI Move Request
// ============================================================

async function requestAIMove() {
    if (game.game_over() || !gameInProgress) return;

    setThinking(true);
    updateStatus('AI is thinking...', 'waiting');

    try {
        const currentFen = game.fen();

        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), thinkTime + 6000);

        const response = await fetch(`${API_BASE}/move`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            signal: controller.signal,
            body: JSON.stringify({
                fen: currentFen,
                moves: '',
                timeMs: thinkTime
            })
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const data = await response.json();

        if (data.success && data.bestMove) {
            if (data.bestMove === '0000') {
                console.warn('Engine reported no move (0000). Model may be missing.');
                updateStatus('AI is offline: model does not exist', 'ended');
                return;
            }

            // Parse UCI move
            const from = data.bestMove.substring(0, 2);
            const to = data.bestMove.substring(2, 4);
            const promotion = data.bestMove.length > 4 ? data.bestMove[4] : undefined;

            const move = game.move({
                from: from,
                to: to,
                promotion: promotion
            });

            if (move) {
                board.position(game.fen());
                addMoveToHistory(move);
                updateGameStatus();
                updateCheckHighlight();

                if (game.game_over()) {
                    handleGameOver();
                    return;
                }

                updateStatus('Your turn', 'active');
            } else {
                console.error('Engine returned illegal move for current position:', data.bestMove);
                updateStatus('AI is offline: model does not exist', 'ended');
            }
        } else {
            console.error('Engine error:', data.error);
            updateStatus('Engine error — try again', 'ended');
        }
    } catch (error) {
        console.error('Failed to reach API:', error);
        updateStatus('Cannot reach server', 'ended');

        // Fallback: make a random move
        makeRandomAIMove();
    } finally {
        setThinking(false);
    }
}

function makeRandomAIMove() {
    const moves = game.moves();
    if (moves.length === 0) return;

    const move = game.move(moves[Math.floor(Math.random() * moves.length)]);
    board.position(game.fen());
    addMoveToHistory(move);
    updateGameStatus();
    updateCheckHighlight();
    updateStatus('Your turn (offline mode)', 'active');
}

// ============================================================
// Move History
// ============================================================

function addMoveToHistory(move) {
    moveHistory.push(move);
    renderMoveHistory();
}

function renderMoveHistory() {
    const container = document.getElementById('moveList');

    if (moveHistory.length === 0) {
        container.innerHTML = '<p class="empty-message">No moves yet. Make your first move!</p>';
        return;
    }

    let html = '';
    for (let i = 0; i < moveHistory.length; i += 2) {
        const num = Math.floor(i / 2) + 1;
        const whiteMove = moveHistory[i].san;
        const blackMove = (i + 1 < moveHistory.length) ? moveHistory[i + 1].san : '';

        html += `<div class="move-row">
            <span class="move-number">${num}.</span>
            <span class="move-white">${whiteMove}</span>
            <span class="move-black">${blackMove}</span>
        </div>`;
    }

    container.innerHTML = html;
    container.scrollTop = container.scrollHeight;
}

// ============================================================
// Game Controls
// ============================================================

function newGame() {
    game.reset();
    board.start();
    moveHistory = [];
    gameInProgress = true;
    renderMoveHistory();
    updateCheckHighlight();
    updateStatus('Your turn', 'active');

    playerColor = document.getElementById('playAs').value;
    thinkTime = parseInt(document.getElementById('thinkTime').value);

    // Orient board
    board.orientation(playerColor);

    // If playing as black, request AI move first
    if (playerColor === 'black') {
        setTimeout(requestAIMove, 500);
    }

    // Notify backend
    fetch(`${API_BASE}/newgame`, { method: 'POST' }).catch(() => {});
}

function flipBoard() {
    board.flip();
}

function undoMove() {
    if (moveHistory.length < 2) return;

    game.undo();
    game.undo();
    moveHistory.pop();
    moveHistory.pop();

    board.position(game.fen());
    renderMoveHistory();
    updateCheckHighlight();
    updateStatus('Your turn', 'active');
}

// ============================================================
// UI Helpers
// ============================================================

function setThinking(active) {
    const indicator = document.getElementById('thinkingIndicator');
    if (active) {
        indicator.classList.add('active');
    } else {
        indicator.classList.remove('active');
    }
}

function updateStatus(text, type) {
    const container = document.getElementById('gameStatus');
    if (!container) {
        console.warn('Missing #gameStatus element');
        return;
    }
    container.innerHTML = `<span class="status-badge ${type}">${text}</span>`;
}

function updateGameStatus() {
    if (game.in_checkmate()) {
        const winner = game.turn() === 'w' ? 'Black' : 'White';
        updateStatus(`Checkmate! ${winner} wins`, 'ended');
    } else if (game.in_stalemate()) {
        updateStatus('Stalemate — Draw', 'ended');
    } else if (game.in_draw()) {
        updateStatus('Draw', 'ended');
    } else if (game.in_check()) {
        updateStatus('Check!', 'waiting');
    }
}

function handleGameOver() {
    gameInProgress = false;
    setThinking(false);
    updateGameStatus();
}

// ============================================================
// Engine Status Check
// ============================================================

async function checkEngineStatus() {
    const dot = document.querySelector('.status-dot');
    const text = document.querySelector('.status-text');

    try {
        const response = await fetch(`${API_BASE}/status`);
        const data = await response.json();

        if (data.initialized) {
            dot.classList.add('connected');
            dot.classList.remove('error');
            text.textContent = 'Engine Online';
        } else {
            dot.classList.remove('connected');
            dot.classList.add('error');
            text.textContent = 'Engine Loading...';
        }
    } catch {
        dot.classList.remove('connected');
        dot.classList.add('error');
        text.textContent = 'Offline (random moves)';
    }
}

// ============================================================
// Settings change handlers
// ============================================================

document.getElementById('playAs').addEventListener('change', (e) => {
    playerColor = e.target.value;
});

document.getElementById('thinkTime').addEventListener('change', (e) => {
    thinkTime = parseInt(e.target.value);
});

// ============================================================
// Init
// ============================================================

$(document).ready(function() {
    initBoard();
    checkEngineStatus();
    setInterval(checkEngineStatus, 10000);
    updateStatus('Your turn', 'active');
});
