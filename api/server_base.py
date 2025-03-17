from fastapi import FastAPI
from pydantic import BaseModel
from wtpsplit import SaT
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch
from utils import *
import timeit

device = get_device()
model = SaT("sat-12l-sm",
            from_pretrained_kwargs = {
                "torch_dtype": torch.float32,
            })
model.half().to(device)
text_splitter = RecursiveCharacterTextSplitter(  # split theo doạn văn
        chunk_size=500,
        chunk_overlap=0,
        separators=["\n\n", "\n", ],  # Ưu tiên cắt theo đoạn văn
)

app = FastAPI()

class TextData(BaseModel):
        text: str


@app.post("/chunk-text/")
async def process_text(data: TextData):
        sub_text = text_splitter.split_text(data.text)

        start_time = timeit.default_timer()
        sub_text = chunking(
            model,
            sub_text,
            60)
        end_time = timeit.default_timer()
        execution_time = end_time - start_time

        return {
            "chunks": sub_text,
            "total_chunk": len(sub_text),
            "execution_time": f"{execution_time:.5f}",
        }