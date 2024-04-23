import streamlit as st

from dashboards.bpe_review import main as bpe_review
from dashboards.quantizer_review import main as quantizer_review
from dashboards.tokenizer_review import main as tokenizer_review
from dashboards.awesome_tokenizer_review import main as awesome_tokenizer_review


def main():
    dashboards = ["tokenizer review", "quantizer review", "bpe_review", "awesome_tokenizer_review"]
    display_mode = st.selectbox("display mode", options=dashboards)

    match display_mode:
        case "quantizer review":
            quantizer_review()
        case "tokenizer review":
            tokenizer_review()
        case "bpe_review":
            bpe_review()
        case "awesome_tokenizer_review":
            awesome_tokenizer_review()


if __name__ == "__main__":
    main()
