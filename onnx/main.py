from fastapi import FastAPI
from pydantic import BaseModel
from wtpsplit import SaT
from langchain.text_splitter import RecursiveCharacterTextSplitter
from itertools import chain

model = SaT("sat-12l-sm")
model.half().to("cuda")
text_splitter = RecursiveCharacterTextSplitter(  # split theo doạn văn
        chunk_size=500,
        chunk_overlap=0,
        separators=["\n\n", "\n", ],  # Ưu tiên cắt theo đoạn văn
)


def chunking(sub_text: list[str], number: int = 128) -> list[str]:
        """
        Split the given sub_text into chunks using the SaT model.

        Args:
            sub_text (list[str]): List of text segments to be split.
            number (int): Block size for splitting. Default is 128.

        Returns:
            list[str]: List of split text segments.
        """

        new_sub_text = model.split(sub_text,
                                   block_size=number,
                                   verbose=False)
        new_sub_text = list(chain.from_iterable(list(new_sub_text)))
        return list(new_sub_text)


app = FastAPI()


class TextData(BaseModel):
        text: str


@app.post("/chunk-text/")
async def process_text(data: TextData):
        sub_text = text_splitter.split_text(data.text)
        sub_text = chunking(
            sub_text,
            256)

        return {"received_text": sub_text,
                "total_chunk": len(sub_text)}
