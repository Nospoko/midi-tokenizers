import streamlit as st

from dashboards.quantizer_review import main as quantizer_review
from dashboards.tokenizer_review import main as tokenizer_review


def main():
    display_mode = st.selectbox("display mode", options=["tokenizer review", "quantizer review"])

    match display_mode:
        case "quantizer review":
            quantizer_review()
        case "tokenizer review":
            tokenizer_review()


if __name__ == "__main__":
    main()
