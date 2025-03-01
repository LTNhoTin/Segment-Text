from itertools import chain
from wtpsplit import SaT
import torch


def chunking(model: SaT, sub_text: list[str], number: int = 128) -> list[str]:
    """
    Split the given sub_text into chunks using the SaT model.

    Args:
        model (SaT): The SaT model used for splitting the text.
        sub_text (list[str]): List of text segments to be split.
        number (int): Block size for splitting. Default is 128.

    Returns:
        list[str]: List of split text segments.
    """

    new_sub_text = model.split(sub_text,
                               block_size=number,
                               strip_whitespace=True,
                               verbose=False)
    new_sub_text = list(chain.from_iterable(list(new_sub_text)))
    return list(new_sub_text)

def device():
    """
    Check if CUDA is available and return appropriate device.

    Returns:
        str: 'cuda' if available, otherwise 'cpu'
    """
    return 'cuda' if torch.cuda.is_available() else 'cpu'