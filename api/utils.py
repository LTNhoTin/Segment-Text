from itertools import chain
from wtpsplit import SaT
import torch
from transformers import AutoTokenizer

deepseek_tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")


def chunking_by_tokenizer(sub_texts: list,
                          max_length: int,
                          model: SaT,
                          tokenizer=deepseek_tokenizer):

    # Use the SaT model to split the text
    sub_texts = model.split(sub_texts,
                                block_size=128,
                                strip_whitespace=True,
                                verbose=False)
    sub_texts = list(chain.from_iterable(list(sub_texts)))
    id_sub_texts = [tokenizer.encode(text, add_special_tokens=False) for text in sub_texts]

    new_sub_texts = []
    id_chunk = []
    for ids in id_sub_texts:
        check_chunk = id_chunk + ids
        n_id = len(check_chunk)
        if n_id > max_length:
            new_sub_texts.append(tokenizer.decode(id_chunk))
            id_chunk = ids
        else:
            id_chunk += ids

    if len(id_chunk) > 0:
        new_sub_texts.append(tokenizer.decode(id_chunk))

    return new_sub_texts


def chunking(model: SaT, sub_text: list[str], number: int = 128) -> list[str]:
    """
    Splits the input text into smaller chunks based on the specified number of tokens.

    Args:
        model (SaT): The SaT model used for splitting the text.
        sub_text (list[str]): The list of strings to be chunked.
        number (int, optional): The maximum number of tokens per chunk. Defaults to 128.

    Returns:
        list[str]: A list of chunked text strings.
    """

    # Use the SaT model to split the text
    sub_text = model.split(sub_text,
                           block_size=128,
                           strip_whitespace=True,
                           verbose=False)
    sub_text = list(chain.from_iterable(list(sub_text)))

    # print(sub_text)

    # Group the text into chunks based on the specified number of tokens
    new_sub_text = []
    chunk = ""

    for text in sub_text:
        check_chunk = chunk + " " + text
        n_token = len(check_chunk.split())
        if n_token > number:
            new_sub_text.append(chunk)
            chunk = text
        else:
            chunk = chunk + " " + text

    if len(chunk.split()) > 0:
        new_sub_text.append(chunk)

    return new_sub_text


def device() -> str:
    """
    Check if CUDA is available and return appropriate device.

    Returns:
        str: 'cuda' if available, otherwise 'cpu'
    """
    return 'cuda' if torch.cuda.is_available() else 'cpu'
