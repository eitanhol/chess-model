import os
import glob
from flask import Flask, request, jsonify, render_template_string
from inference import load_model_and_vocab, predict_top_k

app = Flask(__name__)

# --- CONFIG ---
CHECKPOINT_FOLDER = "checkpoints"

# --- GLOBAL STATE ---
model = None
move2id = None
id2move = None
device = None

# --- EMBEDDED HTML/JS FRONTEND ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en" data-bs-theme="dark">
<head>
    <meta charset="UTF-8">
    <title>Chess AI Controller</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
    <link rel="stylesheet" href="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script src="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/chess.js/0.10.3/chess.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>

    <style>
        /* --- DARK THEME OVERRIDES --- */
        body { 
            background-color: #212529; 
            color: #f8f9fa; 
            padding-top: 15px; 
            overflow-x: hidden; 
        }
        
        /* Input Visibility */
        .form-control, .form-select, input, textarea, select {
            background-color: #2b3035 !important;
            color: #ffffff !important;
            border-color: #495057;
        }
        
        .form-control:focus, .form-select:focus {
            border-color: #86b7fe;
            box-shadow: 0 0 0 0.25rem rgba(13, 110, 253, 0.25);
        }
        
        /* Headers & Spacing */
        .card-header { font-weight: bold; color: #e0e0e0 !important; }
        h2 { margin-bottom: 1.5rem !important; color: #f8f9fa; }
        .card { background-color: #343a40; border-color: #495057; margin-bottom: 1rem; }
        .list-group-item { background-color: #343a40; color: white; border-color: #495057; }
        
        /* --- BOARD SIZING --- */
        #boardContainer {
            position: relative; 
            width: 100%; 
            max-width: 650px; 
            margin: auto;
        }

        /* --- SPARE PIECES --- */
        div[class*="spare-pieces-"] {
            padding: 0 !important;
            margin: 0 !important;
            height: 50px !important; 
            display: flex;
            justify-content: center;
            align-items: center;
        }
        div[class*="spare-pieces-"] img {
            width: 45px !important;
            height: 45px !important;
            cursor: grab;
        }
        
        .chessboard-63f37 img { opacity: 1 !important; }
        
        /* Arrow Overlay */
        .arrow-overlay {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none; 
            z-index: 1000;
        }
        .arrow-line { stroke: #ffc107; stroke-linecap: round; }
        .arrow-head { fill: #ffc107; }

        /* Disabled State */
        .disabled-section {
            opacity: 0.4;
            pointer-events: none; 
            filter: grayscale(80%);
        }
        
        .is-invalid-fen { border: 2px solid #dc3545 !important; }
        
        /* Predictions Box */
        #predCard .card-body {
            max-height: calc(100vh - 180px); 
            overflow-y: auto;
        }
        
        /* Custom Radio Styling */
        .btn-check:checked + .btn-outline-warning {
            background-color: #ffc107;
            color: #000;
        }
        .btn-check:checked + .btn-outline-success {
            background-color: #198754;
            color: #fff;
        }

        /* Promotion Modal Buttons */
        .promo-btn {
            font-size: 2rem;
            width: 60px;
            height: 60px;
            margin: 5px;
        }
    </style>
</head>
<body>

<div class="modal fade" id="promotionModal" tabindex="-1" data-bs-backdrop="static" data-bs-keyboard="false">
  <div class="modal-dialog modal-sm modal-dialog-centered">
    <div class="modal-content bg-dark border-secondary">
      <div class="modal-header border-secondary">
        <h5 class="modal-title text-white">Promote to:</h5>
      </div>
      <div class="modal-body text-center">
        <button class="btn btn-outline-light promo-btn" onclick="commitPromotion('q')"><i class="fas fa-chess-queen"></i></button>
        <button class="btn btn-outline-light promo-btn" onclick="commitPromotion('r')"><i class="fas fa-chess-rook"></i></button>
        <button class="btn btn-outline-light promo-btn" onclick="commitPromotion('b')"><i class="fas fa-chess-bishop"></i></button>
        <button class="btn btn-outline-light promo-btn" onclick="commitPromotion('n')"><i class="fas fa-chess-knight"></i></button>
      </div>
      <div class="modal-footer border-secondary justify-content-center">
        <button type="button" class="btn btn-secondary btn-sm" onclick="cancelPromotion()">Cancel</button>
      </div>
    </div>
  </div>
</div>

<div class="container-fluid px-4">
    <h2 class="text-center">Neural Chess Interface</h2>
    
    <div class="row">
        <div class="col-md-3">
            <div class="card">
                <div class="card-header">Settings</div>
                <div class="card-body">
                    <div class="input-group mb-3">
                        <select id="modelSelect" class="form-select">
                            {% for m in models %}
                            <option value="{{ m }}">{{ m }}</option>
                            {% endfor %}
                        </select>
                        <button class="btn btn-primary" onclick="loadModel()">Load</button>
                    </div>
                    <div id="loadStatus" class="small text-success mb-2 fw-bold" style="min-height: 20px;"></div>

                    <div id="settingsContent" class="disabled-section">
                        <div class="row g-2 align-items-end mb-3">
                            <div class="col-4">
                                <label class="small">Elo:</label>
                                <input type="number" id="elo" class="form-control" value="1200" min="1000" max="3000" step="50" onchange="clampElo()">
                            </div>
                            <div class="col-4">
                                <label class="small">Top-k:</label>
                                <input type="number" id="topk" class="form-control" value="3" min="1" max="10">
                            </div>
                            <div class="col-4">
                                <button class="btn btn-primary w-100" onclick="refreshView()">Apply</button>
                            </div>
                        </div>

                        <div class="d-flex justify-content-between mb-3">
                            <div class="form-check form-switch">
                                <input class="form-check-input" type="checkbox" id="showPredictions" checked onchange="refreshView()">
                                <label class="form-check-label" for="showPredictions">Move List</label>
                            </div>
                            <div class="form-check form-switch">
                                <input class="form-check-input" type="checkbox" id="showArrows" checked onchange="refreshView()">
                                <label class="form-check-label" for="showArrows">Arrows</label>
                            </div>
                        </div>
                        
                        <label class="small mb-1">Game Mode</label>
                        <div class="btn-group w-100 mb-3" role="group">
                            <input type="radio" class="btn-check" name="modeRadio" id="modeAnalysis" autocomplete="off" checked onchange="handleModeChange()">
                            <label class="btn btn-outline-warning" for="modeAnalysis">Analysis</label>

                            <input type="radio" class="btn-check" name="modeRadio" id="modePlay" autocomplete="off" onchange="handleModeChange()">
                            <label class="btn btn-outline-success" for="modePlay">Play Bot</label>
                        </div>

                        <label id="sideLabel" class="small mb-1">Analysis: Predict For</label>
                        <div class="btn-group w-100 mb-3" role="group">
                            <input type="radio" class="btn-check" name="sideRadio" id="sideWhite" autocomplete="off" checked onchange="handleSideChange()">
                            <label class="btn btn-outline-light" for="sideWhite">White</label>

                            <input type="radio" class="btn-check" name="sideRadio" id="sideBlack" autocomplete="off" onchange="handleSideChange()">
                            <label class="btn btn-outline-light" for="sideBlack">Black</label>
                        </div>
                        
                        <button class="btn btn-outline-light w-100 mb-2" onclick="makeAiMove()">Force AI Move</button>
                        
                        <button class="btn btn-secondary w-100" onclick="resetCurrentBoard()">Reset Board</button>
                    </div>
                </div>
            </div>

            <div id="pgnCard" class="card disabled-section">
                <div class="card-header">PGN (Edit & Apply)</div>
                <div class="card-body">
                    <textarea id="pgnInput" class="form-control mb-2" rows="3" placeholder="1. e4 e5 ..."></textarea>
                    <button class="btn btn-primary w-100 mb-2" onclick="loadPgnText()">Apply PGN</button>
                    <input type="file" id="pgnFile" class="form-control" accept=".pgn">
                </div>
            </div>
        </div>

        <div id="boardPanel" class="col-md-6 disabled-section">
            <div id="boardContainer">
                <div id="myBoard"></div>
            </div>
            
            <div class="text-center mt-3">
                <button class="btn btn-secondary btn-nav" onclick="navStart()"><i class="fas fa-angle-double-left"></i></button>
                <button class="btn btn-secondary btn-nav" onclick="navPrev()"><i class="fas fa-angle-left"></i></button>
                <button class="btn btn-secondary btn-nav" onclick="navNext()"><i class="fas fa-angle-right"></i></button>
                <button class="btn btn-secondary btn-nav" onclick="navEnd()"><i class="fas fa-angle-double-right"></i></button>
            </div>

            <div class="mt-3 px-3">
                <div class="input-group">
                    <input type="text" id="fenDisplay" class="form-control text-center" placeholder="FEN String">
                    <button class="btn btn-outline-secondary" onclick="setFromFen()">Set FEN</button>
                </div>
            </div>
        </div>

        <div class="col-md-3">
             <div id="predCard" class="card disabled-section">
                <div class="card-header">AI Predictions <span id="turnIndicator" class="badge bg-secondary float-end"></span></div>
                <div class="card-body">
                    <div id="thinking" class="text-warning fw-bold mb-2" style="display:none;">Thinking...</div>
                    <ul id="predList" class="list-group list-group-flush"></ul>
                </div>
            </div>
        </div>
    </div>
</div>

<script>
    var board = null;
    
    // --- SEPARATE GAME STATES ---
    var gameAnalysis = new Chess();
    var gamePlay = new Chess();
    
    var currentGame = gameAnalysis; 
    var currentMode = 'analysis'; 
    var redoStack = []; 
    var lastPredictions = [];
    
    // --- DEBOUNCE TIMER ---
    var predictionDebounce = null;
    
    // --- PROMOTION STATE ---
    var pendingPromotion = null; // { source: 'e7', target: 'e8' }
    var promoModal = null;
    
    // --- HELPER: CLAMP ELO ---
    function clampElo() {
        let eloInput = document.getElementById('elo');
        let val = parseInt(eloInput.value);
        if (val < 1000) eloInput.value = 1000;
        if (val > 3000) eloInput.value = 3000;
    }

    // --- ENABLE INTERFACE ---
    function enableInterface() {
        $('#settingsContent').removeClass('disabled-section');
        $('#pgnCard').removeClass('disabled-section');
        $('#predCard').removeClass('disabled-section');
        $('#boardPanel').removeClass('disabled-section');
    }

    // --- ARROW VISUALIZATION ---
    function clearArrows() { $('#arrowOverlay').remove(); }

    function renderArrows(predictions) {
        clearArrows();
        if (!$('#showArrows').is(':checked') || !predictions || predictions.length === 0) return;

        var $overlay = $(`
            <svg id="arrowOverlay" class="arrow-overlay" viewBox="0 0 8 8">
                <defs></defs>
            </svg>
        `);
        
        var $defs = $overlay.find('defs');
        var orientation = board.orientation(); 
        
        predictions.forEach(function(pred, index) {
            var width = 0.15;
            // Opacity = Probability (Clamped between 0.15 and 0.90)
            var opacity = Math.max(0.15, Math.min(0.90, pred.prob));
            
            var uniqueId = "arrowhead-" + index;
            
            var $marker = $(document.createElementNS("http://www.w3.org/2000/svg", "marker"));
            $marker.attr({
                id: uniqueId,
                markerWidth: "3",
                markerHeight: "3",
                refX: "2",
                refY: "1.5",
                orient: "auto"
            });
            
            var $poly = $(document.createElementNS("http://www.w3.org/2000/svg", "polygon"));
            $poly.attr({
                points: "0 0, 3 1.5, 0 3",
                fill: "#ffc107",
                "fill-opacity": opacity 
            });
            
            $marker.append($poly);
            $defs.append($marker);

            // Draw line
            var uci = pred.uci;
            var coords = getCoords(uci.substring(0, 2), uci.substring(2, 4), orientation);

            var $line = $(document.createElementNS("http://www.w3.org/2000/svg", "line"));
            $line.attr({
                x1: coords.x1, y1: coords.y1,
                x2: coords.x2, y2: coords.y2,
                'stroke-opacity': opacity,
                'stroke-width': width,
                'marker-end': `url(#${uniqueId})`
            }).addClass('arrow-line');
            
            $overlay.append($line);
        });

        $('#myBoard').append($overlay);
    }

    function getCoords(source, target, orientation) {
        var fileMap = { 'a':0, 'b':1, 'c':2, 'd':3, 'e':4, 'f':5, 'g':6, 'h':7 };
        var srcFile = fileMap[source[0]];
        var srcRank = parseInt(source[1]) - 1; 
        var tgtFile = fileMap[target[0]];
        var tgtRank = parseInt(target[1]) - 1;

        if (orientation === 'white') {
            return { x1: srcFile + 0.5, y1: 7 - srcRank + 0.5, x2: tgtFile + 0.5, y2: 7 - tgtRank + 0.5 };
        } else {
            return { x1: 7 - srcFile + 0.5, y1: srcRank + 0.5, x2: 7 - tgtFile + 0.5, y2: tgtRank + 0.5 };
        }
    }

    // --- DRAG & DROP LOGIC ---
    function onDragStart (source, piece, position, orientation) {
        if ($('#boardPanel').hasClass('disabled-section')) return false; 
        if (source === 'spare' && currentMode === 'analysis') return true;
        
        if (currentMode !== 'analysis') {
            if (currentGame.game_over()) return false;
            // Check if piece color matches player color
            if (currentMode === 'play_white' && piece.search(/^b/) !== -1) return false;
            if (currentMode === 'play_black' && piece.search(/^w/) !== -1) return false;
        }
        return true;
    }

    function onDrop (source, target, piece, newPos, oldPos, orientation) {
        if (source === target) return 'snapback';

        // 1. Analysis Mode
        if (currentMode === 'analysis') {
            if (target === 'offboard') {
                if (source !== 'spare') {
                    currentGame.remove(source);
                    updateStatus(true);
                    return 'trash';
                }
                return 'snapback';
            }

            // In analysis, we just set the board state blindly to what the user dragged
            setTimeout(function() {
                var boardFen = Chessboard.objToFen(newPos);
                var currentTurn = currentGame.turn(); 
                var components = currentGame.fen().split(" ");
                var suffix = components.slice(1).join(" "); 
                var newFen = boardFen + " " + suffix;

                var valid = currentGame.load(newFen);
                if (!valid) {
                    newFen = boardFen + " " + currentTurn + " - - 0 1";
                    valid = currentGame.load(newFen);
                }

                if (valid) {
                    $('#fenDisplay').removeClass('is-invalid-fen');
                    $('#pgnInput').val('[SetUp "1"]\\n[FEN "' + newFen + '"]'); 
                    updateStatus(true); 
                } else {
                    $('#fenDisplay').addClass('is-invalid-fen');
                }
            }, 50);
            return;
        }

        // 2. Play Mode (with Promotion Check)
        
        // A) Test if this is a legal move assuming Queen promotion
        var testMove = currentGame.move({ from: source, to: target, promotion: 'q' });
        
        // If illegal even with promotion, snapback
        if (testMove === null) return 'snapback';

        // B) Check if it WAS a promotion move
        if (testMove.flags.includes('p')) {
            // It is a promotion! Undo the test move immediately.
            currentGame.undo();
            
            // Save state and show modal
            pendingPromotion = { source: source, target: target };
            promoModal.show();
            
            // Return 'snapback' so the pawn returns to start square visually 
            // until the user picks a piece in the modal
            return 'snapback'; 
        }

        // C) Not a promotion, logic proceeds as normal
        redoStack = [];
        $('#pgnInput').val(currentGame.pgn());
        updateStatus();

        if (currentMode.startsWith('play') && !currentGame.game_over()) {
            makeAiMove();
        }
    }

    // --- PROMOTION HANDLING ---
    function commitPromotion(pieceType) {
        if (!pendingPromotion) return;
        
        promoModal.hide();
        var move = currentGame.move({
            from: pendingPromotion.source,
            to: pendingPromotion.target,
            promotion: pieceType
        });
        
        pendingPromotion = null;
        
        if (move) {
            board.position(currentGame.fen());
            redoStack = [];
            $('#pgnInput').val(currentGame.pgn());
            updateStatus();
            
            if (currentMode.startsWith('play') && !currentGame.game_over()) {
                makeAiMove();
            }
        }
    }

    function cancelPromotion() {
        pendingPromotion = null;
        promoModal.hide();
        // Board is already snapped back, nothing to do
    }

    function onSnapEnd () {
        if(currentMode !== 'analysis') {
            board.position(currentGame.fen());
        }
    }

    // --- BOARD INIT ---
    function initBoard() {
        var config = {
            draggable: true,
            position: 'start',
            onDragStart: onDragStart,
            onDrop: onDrop,
            onSnapEnd: onSnapEnd,
            pieceTheme: 'https://chessboardjs.com/img/chesspieces/wikipedia/{piece}.png'
        }

        if (currentMode === 'analysis') {
            config.sparePieces = true; 
            config.dropOffBoard = 'trash';
            config.position = gameAnalysis.fen();
        } else {
            config.sparePieces = false;
            config.dropOffBoard = 'snapback';
            config.position = gamePlay.fen();
        }

        board = Chessboard('myBoard', config);
        
        if (currentMode === 'play_black') board.orientation('black');
        else board.orientation('white');
        
        updateStatus();
        $('#pgnInput').val(currentGame.pgn());
    }

    // --- PGN HANDLING ---
    $('#pgnFile').on('change', function(e) {
        var file = e.target.files[0];
        if (!file) return;
        var reader = new FileReader();
        reader.onload = function(e) {
            $('#pgnInput').val(e.target.result);
        };
        reader.readAsText(file);
    });

    function loadPgnText() {
        var pgn = $('#pgnInput').val();
        var success = currentGame.load_pgn(pgn);
        if (!success) {
            alert("Invalid PGN! Game state not updated.");
            return;
        }
        redoStack = [];
        board.position(currentGame.fen());
        updateStatus();
    }
    
    // --- NAVIGATION ---
    function navPrev() {
        var move = currentGame.undo();
        if (move) {
            redoStack.push(move);
            board.position(currentGame.fen());
            $('#pgnInput').val(currentGame.pgn());
            updateStatus();
        }
    }
    
    function navNext() {
        var move = redoStack.pop();
        if (move) {
            currentGame.move(move);
            board.position(currentGame.fen());
            $('#pgnInput').val(currentGame.pgn());
            updateStatus();
        }
    }

    function navStart() {
        currentGame.reset();
        redoStack = []; 
        board.start();
        $('#pgnInput').val(currentGame.pgn());
        updateStatus();
    }
    
    function navEnd() {
        while (redoStack.length > 0) {
            var move = redoStack.pop();
            currentGame.move(move);
        }
        board.position(currentGame.fen());
        $('#pgnInput').val(currentGame.pgn());
        updateStatus();
    }

    // --- API CALLS ---
    function loadModel() {
        var name = $('#modelSelect').val();
        $('#loadStatus').text("Loading...").removeClass('text-success').addClass('text-warning');
        $.ajax({
            url: '/load_model',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ filename: name }),
            success: function(res) {
                if(res.status === 'success') {
                    $('#loadStatus').text("Model Ready!").removeClass('text-warning').addClass('text-success');
                    enableInterface();
                    refreshView();
                }
                else alert(res.message);
            },
            error: function() {
                $('#loadStatus').text("Error connecting to server.").removeClass('text-success').addClass('text-danger');
            }
        });
    }

    function makeAiMove() {
        clampElo(); 
        var sentFen = currentGame.fen();
        console.log("[DEBUG] Starting AI Move. Current FEN:", sentFen);

        $.ajax({
            url: '/predict',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ fen: sentFen, rating: $('#elo').val(), k: 10 }),
            success: function(res) {
                console.log("[DEBUG] Server response received:", res);

                if (res.status === 'success' && res.preds.length > 0) {
                    var moveExecuted = null;
                    
                    for (var i = 0; i < res.preds.length; i++) {
                        var candidate = res.preds[i];
                        console.log("[DEBUG] Processing candidate #" + (i+1) + ":", candidate);
                        
                        try {
                            // Parse UCI
                            var uci = candidate.uci;
                            var fromSquare = uci.substring(0, 2);
                            var toSquare = uci.substring(2, 4);
                            var promotionPiece = uci.length === 5 ? uci.substring(4, 5) : undefined;
                            
                            var moveObj = {
                                from: fromSquare,
                                to: toSquare,
                                promotion: promotionPiece || 'q' // Default to Queen if missing for check
                            };
                            
                            // Check if move is valid in current state
                            // Note: We use the object format which chess.js likes better
                            moveExecuted = currentGame.move(moveObj);
                            
                            console.log("[DEBUG] Move attempt:", moveObj, "Result:", moveExecuted);
                            
                            if (moveExecuted !== null) {
                                console.log("[DEBUG] VALID MOVE FOUND! " + candidate.uci);
                                break; // Stop looking, we found a move
                            } else {
                                console.warn("[DEBUG] Illegal move rejected by Chess.js:", candidate.uci);
                            }
                        } catch (err) {
                            console.error("[DEBUG] Error parsing candidate:", err);
                        }
                    }

                    if (moveExecuted) {
                        redoStack = [];
                        board.position(currentGame.fen());
                        $('#pgnInput').val(currentGame.pgn());
                        updateStatus();
                    } else {
                        console.error("[DEBUG] AI FAILED: No legal moves found in top " + res.preds.length + " predictions.");
                        alert("Debug: AI could not find a legal move. Check console for details.");
                    }
                } else {
                    console.error("[DEBUG] Server returned success but no predictions, or status error.");
                }
            },
            error: function(xhr, status, error) {
                console.error("[DEBUG] AJAX Error:", status, error);
            }
        });
    }

    function getPredictions(fen) {
        clampElo(); 
        var k = $('#topk').val();
        
        // --- DEBOUNCE LOGIC (100ms) ---
        clearTimeout(predictionDebounce);
        
        // Clear UI immediately to indicate stale state, or leave it?
        // Leaving it is usually smoother visually.
        
        predictionDebounce = setTimeout(function() {
            $.ajax({
                url: '/predict',
                type: 'POST',
                contentType: 'application/json',
                data: JSON.stringify({ fen: fen, rating: $('#elo').val(), k: k }),
                success: function(res) {
                    $('#predList').empty();
                    clearArrows(); 
                    if (res.status === 'success') {
                        lastPredictions = res.preds;
                        
                        if ($('#showPredictions').is(':checked')) {
                            res.preds.forEach(function(p, index) {
                                $('#predList').append(
                                    `<li class="list-group-item">
                                        <strong>${index+1}. ${p.san}</strong> 
                                        <span class="badge bg-primary float-end">${(p.prob*100).toFixed(1)}%</span>
                                    </li>`
                                );
                            });
                        }
                        renderArrows(res.preds);
                    }
                }
            });
        }, 100);
    }

    function refreshView() {
        getPredictions(currentGame.fen());
    }

    // --- MODE HANDLING ---
    
    function handleModeChange() {
        resolveModeAndReset();
    }

    function handleSideChange() {
        var isAnalysis = $('#modeAnalysis').is(':checked');
        if (isAnalysis) {
            var color = $('#sideWhite').is(':checked') ? 'w' : 'b';
            setTurn(color);
        } else {
            resolveModeAndReset();
        }
    }

    function resolveModeAndReset() {
        var isAnalysis = $('#modeAnalysis').is(':checked');
        var isWhite = $('#sideWhite').is(':checked');
        
        if (isAnalysis) {
            $('#sideLabel').text("Analysis: Predict For");
            currentMode = 'analysis';
            currentGame = gameAnalysis;
        } else {
            $('#sideLabel').text("Play as");
            currentMode = isWhite ? 'play_white' : 'play_black';
            currentGame = gamePlay;
        }

        redoStack = [];
        initBoard();

        if (currentMode === 'play_black' && currentGame.turn() === 'w') {
            makeAiMove();
        }
    }

    function resetCurrentBoard() {
        currentGame.reset();
        $('#pgnInput').val(""); 
        redoStack = [];
        lastPredictions = [];
        board.start();
        updateStatus();
    }

    function setFromFen() {
        var fen = $('#fenDisplay').val();
        var valid = currentGame.load(fen);
        if(!valid) { alert("Invalid FEN"); return; }
        redoStack = [];
        board.position(fen);
        $('#pgnInput').val(currentGame.pgn());
        updateStatus();
    }

    function setTurn(color) {
        if (currentMode !== 'analysis') return;
        var fen = currentGame.fen();
        var parts = fen.split(' ');
        if (parts.length > 1 && parts[1] !== color) {
            parts[1] = color;
            var newFen = parts.join(' ');
            var valid = currentGame.load(newFen);
            if (valid) {
                board.position(currentGame.fen());
                updateStatus(); 
            } else {
                console.log("Cannot swap turn: Position would be illegal.");
                updateStatus(); 
            }
        }
    }

    function updateStatus(skipPreds) {
        var fen = currentGame.fen();
        $('#fenDisplay').val(fen);
        
        var turn = currentGame.turn();
        var turnText = turn === 'w' ? "White to Move" : "Black to Move";
        $('#turnIndicator').text(turnText);

        if (currentMode === 'analysis') {
            if (turn === 'w') $('#sideWhite').prop('checked', true);
            else $('#sideBlack').prop('checked', true);
        }

        if (!$('#boardPanel').hasClass('disabled-section') && !skipPreds) {
            if ($('#showPredictions').is(':checked') || $('#showArrows').is(':checked')) {
                if(fen.includes('k') && fen.includes('K')) {
                    getPredictions(fen);
                } else {
                    $('#predList').empty().append('<li class="list-group-item text-danger">Invalid Board</li>');
                    clearArrows();
                }
            } else {
                $('#predList').empty();
                clearArrows();
            }
        }
    }

    $(document).ready(function() {
        // Initialize Bootstrap Modal
        promoModal = new bootstrap.Modal(document.getElementById('promotionModal'));
        initBoard();
    });
</script>
</body>
</html>
"""

# --- FLASK ROUTES ---

@app.route('/')
def index():
    models = []
    if os.path.exists(CHECKPOINT_FOLDER):
        models = [os.path.basename(f) for f in glob.glob(os.path.join(CHECKPOINT_FOLDER, "*.pt"))]
    return render_template_string(HTML_TEMPLATE, models=models)

@app.route('/load_model', methods=['POST'])
def load_model_route():
    global model, move2id, id2move, device
    try:
        data = request.json
        filename = data.get('filename')
        path = os.path.join(CHECKPOINT_FOLDER, filename)
        if not os.path.exists(path):
            return jsonify({"status": "error", "message": "File not found"})
            
        model, move2id, id2move, device = load_model_and_vocab(path)
        return jsonify({"status": "success", "message": f"Loaded {filename}"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/predict', methods=['POST'])
def predict_route():
    global model
    if not model:
        print("[DEBUG] Error: Model not loaded")
        return jsonify({"status": "error", "message": "Model not loaded"})
    
    data = request.json
    print(f"[DEBUG] Predict request received. FEN: {data['fen']}") # <--- DEBUG LOG
    
    try:
        preds = predict_top_k(
            model, id2move, 
            data['fen'], 
            int(data['rating']), 
            int(data['k']), 
            device
        )
        print(f"[DEBUG] Predictions generated: {preds}") # <--- DEBUG LOG
        return jsonify({"status": "success", "preds": preds})
    except Exception as e:
        print(f"[DEBUG] Exception during prediction: {str(e)}") # <--- DEBUG LOG
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    print("STARTING SERVER...")
    print("Please click this link to open: http://127.0.0.1:5000")
    app.run(debug=True, port=5000)