# Segment-Text

> Please press ⭐ button and/or cite papers if you feel helpful.

<div align="center">

[![python](https://img.shields.io/badge/-Python_3.10.11-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![pytorch](https://img.shields.io/badge/Torch_2.0.1-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![cuda](https://img.shields.io/badge/-CUDA_11.8-green?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit-archive)
</div>

## Installation

Create a new conda environment.
```bash
conda create --name chunker python=3.10
conda activate chunker
```
Install the required packages.
```bash
pip install -r requirements.txt
pip install onnxruntime==1.21.0
pip install onnx
# package for onnxruntime gpu support
pip install onnxruntime-gpu==1.21.0
```

## Run
Run pytorch model server at port 8000.
```bash
cd api
uvicorn server_base:app --host 0.0.0.0 --port 8000
```

Run onnx model server at port 8001.
```bash
cd api
uvicorn server_onnx:app --host 0.0.0.0 --port 8001
```
## Testing
Use this command to test the FastAPI server. Can change the text to demo.txt. 
```bash
curl -X 'POST' 'http://127.0.0.1:8000/chunk-text/' \
     -H 'Content-Type: application/json' \
     -d '{"text": "Đây là một đoạn văn mẫu gửi đến server FastAPI."}'

curl -X POST "http://127.0.0.1:8000/chunk-text/" -H "Content-Type: application/json" -d "{\"text\": \"Đây là một đoạn văn mẫu gửi đến server FastAPI.\"}"

curl -X POST "http://127.0.0.1:8000/chunk-text/" ^
     -H "Content-Type: application/json" ^
     --data-binary "@demo.txt"

```

## Export to ONNX

```bash 
python convert/export_to_onnx_sat.py --model_name_or_path=segment-any-text/sat-12l-sm --output_dir=models/sat-12l-sm
```

## Run ONNX model

```bash
docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v $(pwd)/models:/models nvcr.io/nvidia/tritonserver:25.02-py3 \
    tritonserver --model-repository=/models
```
## Perf-Analyser
- Run this command in the terminal to analyze the performance of the model.
```bash

cd /home/thatlq1812/projects/OJT/Segment-Text/perf_results

perf_analyzer -m sat-12l-sm -b 1 -u localhost:8000 --shape input_ids:1,512 attention_mask:1,512 --input-data zero --concurrency-range 1:4 > result_sat_12l_sm.txt
perf_analyzer -m sat-12l-sm-fp32 -b 1 -u localhost:8000 --shape input_ids:1,512 attention_mask:1,512 --input-data zero --concurrency-range 1:4 > result_sat_12l_sm_fp32.txt
perf_analyzer -m sat-12l-sm-fp16 -b 1 -u localhost:8000 --shape input_ids:1,512 attention_mask:1,512 --input-data zero --concurrency-range 1:4 > result_sat_12l_sm_fp16.txt
perf_analyzer -m sat-12l-sm-fp32-opt1 -b 1 -u localhost:8000 --shape input_ids:1,512 attention_mask:1,512 --input-data zero --concurrency-range 1:4 > result_sat_12l_sm_fp32_opt1.txt
perf_analyzer -m sat-12l-sm-fp16-opt1 -b 1 -u localhost:8000 --shape input_ids:1,512 attention_mask:1,512 --input-data zero --concurrency-range 1:4 > result_sat_12l_sm_fp16_opt1.txt
perf_analyzer -m sat-12l-sm-fp32-opt2 -b 1 -u localhost:8000 --shape input_ids:1,512 attention_mask:1,512 --input-data zero --concurrency-range 1:4 > result_sat_12l_sm_fp32_opt2.txt
perf_analyzer -m sat-12l-sm-fp16-opt2 -b 1 -u localhost:8000 --shape input_ids:1,512 attention_mask:1,512 --input-data zero --concurrency-range 1:4 > result_sat_12l_sm_fp16_opt2.txt

perf_analyzer -m sat-12l-sm -b 1 -u localhost:8000 --shape input_ids:1,512 attention_mask:1,512 --input-data zero --concurrency-range 1:8 > result_sat_12l_sm_8.txt
perf_analyzer -m sat-12l-sm-fp32 -b 1 -u localhost:8000 --shape input_ids:1,512 attention_mask:1,512 --input-data zero --concurrency-range 1:8 > result_sat_12l_sm_fp32_8.txt
perf_analyzer -m sat-12l-sm-fp16 -b 1 -u localhost:8000 --shape input_ids:1,512 attention_mask:1,512 --input-data zero --concurrency-range 1:8 > result_sat_12l_sm_fp16_8.txt
perf_analyzer -m sat-12l-sm-fp32-opt1 -b 1 -u localhost:8000 --shape input_ids:1,512 attention_mask:1,512 --input-data zero --concurrency-range 1:8 > result_sat_12l_sm_fp32_opt1_8.txt
perf_analyzer -m sat-12l-sm-fp16-opt1 -b 1 -u localhost:8000 --shape input_ids:1,512 attention_mask:1,512 --input-data zero --concurrency-range 1:8 > result_sat_12l_sm_fp16_opt1_8.txt
perf_analyzer -m sat-12l-sm-fp32-opt2 -b 1 -u localhost:8000 --shape input_ids:1,512 attention_mask:1,512 --input-data zero --concurrency-range 1:8 > result_sat_12l_sm_fp32_opt2_8.txt
perf_analyzer -m sat-12l-sm-fp16-opt2 -b 1 -u localhost:8000 --shape input_ids:1,512 attention_mask:1,512 --input-data zero --concurrency-range 1:8 > result_sat_12l_sm_fp16_opt2_8.txt

```
- Because this model not support multiple batch size, so we need to change the batch size to 1. " -b 1 "
## Note

Các tham số trong phương thức `split` của SaT

- **`text_or_texts`**: Bắt buộc. Văn bản hoặc danh sách các văn bản cần được phân đoạn.
- **`threshold`**: Tùy chọn. Ngưỡng xác suất để quyết định điểm phân đoạn câu. Nếu `None`, sẽ sử dụng ngưỡng mặc định của mô hình.
- **`stride`**: Tùy chọn. Số lượng ký tự để trượt qua mỗi lần phân đoạn. Mặc định là 64.
- **`block_size`**: Tùy chọn. Kích thước khối tối đa cho mỗi lần phân đoạn. Mặc định là 512.
- **`batch_size`**: Tùy chọn. Số lượng mẫu trong mỗi lô khi phân đoạn. Mặc định là 32.
- **`pad_last_batch`**: Tùy chọn. Nếu `True`, lô cuối cùng sẽ được đệm để đủ kích thước `batch_size`. Mặc định là `False`.
- **`weighting`**: Tùy chọn. Phương pháp trọng số, có thể là "uniform" hoặc "hat". Mặc định là "uniform".
- **`remove_whitespace_before_inference`**: Tùy chọn. Nếu `True`, loại bỏ khoảng trắng trước khi phân đoạn. Mặc định là `False`.
- **`outer_batch_size`**: Tùy chọn. Kích thước lô ngoài cùng khi phân đoạn. Mặc định là 1000.
- **`paragraph_threshold`**: Tùy chọn. Ngưỡng xác suất để quyết định điểm phân đoạn đoạn văn. Mặc định là 0.5.
- **`strip_whitespace`**: Tùy chọn. Nếu `True`, loại bỏ khoảng trắng ở đầu và cuối câu sau khi phân đoạn. Mặc định là `False`.
- **`do_paragraph_segmentation`**: Tùy chọn. Nếu `True`, thực hiện phân đoạn đoạn văn. Mặc định là `False`.
- **`split_on_input_newlines`**: Tùy chọn. Nếu `True`, phân đoạn dựa trên các dòng mới trong đầu vào. Mặc định là `True`.
- **`treat_newline_as_space`**: Đã bị loại bỏ. Sử dụng `split_on_input_newlines` thay thế.
- **`verbose`**: Tùy chọn. Nếu `True`, hiển thị thông tin chi tiết trong quá trình phân đoạn. Mặc định là `False`.