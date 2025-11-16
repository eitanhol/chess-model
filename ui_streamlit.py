# ui_streamlit.py

import streamlit as st
import chess
import chess.svg

from inference import load_model_and_vocab, predict_top_k


@st.cache_resource
def load_model_cached(checkpoint_path: str):
    return load_model_and_vocab(checkpoint_path)


def render_board(fen: str, last_uci: str | None = None):
    board = chess.Board(fen)
    lastmove = None
    if last_uci is not None:
        try:
            lastmove = chess.Move.from_uci(last_uci)
        except ValueError:
            lastmove = None

    svg = chess.svg.board(board, lastmove=lastmove, size=400)
    # Render raw SVG in Streamlit
    st.write(svg, unsafe_allow_html=True)


def main():
    st.title("Rating-Conditioned Human Move Model (Chess)")

    st.sidebar.header("Model settings")
    checkpoint_path = st.sidebar.text_input(
        "Checkpoint path",
        value="checkpoints/rapid_1000_1100_double.pt",
    )

    if not checkpoint_path:
        st.warning("Please specify a checkpoint path.")
        return

    try:
        model, move2id, id2move, device = load_model_cached(checkpoint_path)
    except Exception as e:
        st.error(f"Error loading checkpoint: {e}")
        return

    st.subheader("Position & rating")

    default_fen = chess.STARTING_FEN
    fen = st.text_input("FEN string", value=default_fen)

    col1, col2 = st.columns(2)
    with col1:
        rating = st.number_input("Player rating (Elo)", min_value=800, max_value=2800, value=1050, step=50)
    with col2:
        k = st.number_input("Top-k moves", min_value=1, max_value=10, value=5, step=1)

    if st.button("Predict moves"):
        # Show board first
        st.markdown("### Position")
        render_board(fen)

        # Run model
        try:
            preds = predict_top_k(
                model=model,
                id2move=id2move,
                fen=fen,
                rating=int(rating),
                k=int(k),
                device=device,
            )
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            return

        if not preds:
            st.info("No moves predicted (something went wrong with vocab/FEN).")
            return

        # Highlight top-1 move on a separate board
        best = preds[0]
        st.markdown(f"### Top move highlight: **{best['san']}** ({best['uci']})")
        render_board(fen, last_uci=best["uci"])

        st.markdown("### Top-k predicted moves")
        for i, p in enumerate(preds, start=1):
            st.write(f"{i}. **{p['san']}**  `{p['uci']}`   (p = {p['prob']:.3f})")


if __name__ == "__main__":
    main()
