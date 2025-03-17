from dataclasses import dataclass
from pathlib import Path

import adapters  # noqa
import onnx
import torch
from adapters.models import MODEL_MIXIN_MAPPING  # noqa
from adapters.models.bert.mixin_bert import BertModelAdaptersMixin  # noqa
from huggingface_hub import hf_hub_download, HfApi
from onnxruntime.transformers.optimizer import optimize_model  # noqa
from transformers import AutoModelForTokenClassification, HfArgumentParser

import wtpsplit  # noqa
import wtpsplit.models  # noqa
from wtpsplit.utils import Constants

MODEL_MIXIN_MAPPING["SubwordXLMRobertaModel"] = BertModelAdaptersMixin


@dataclass
class Args:
    model_name_or_path: str = "segment-any-text/sat-12l-sm"  # model from HF Hub: https://huggingface.co/segment-any-text
    output_dir: str = "models/sat-12l-sm-fp32-opt2"  # output directory, saves to current directory
    device: str = "cuda"
    use_lora: bool = False
    lora_path: str = None  # local path to lora weights
    # otherwise, fetch from HF Hub:
    style_or_domain: str = "ud"
    language: str = "en"
    upload_to_hub: bool = False
    use_fp16: bool = False # OPTINAL: Convert model to FP16 or FP32
    # opt_level_value = 1 # from 0 to 3

if __name__ == "__main__":
    (args,) = HfArgumentParser([Args]).parse_args_into_dataclasses()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    model = AutoModelForTokenClassification.from_pretrained(args.model_name_or_path, force_download=False)

    model = model.to(args.device)

    # fetch config.json from huggingface hub
    hf_hub_download(
        repo_id=args.model_name_or_path,
        filename="config.json",
        local_dir=output_dir,
    )

    # LoRA SETUP
    if args.use_lora:
        # adapters need xlm-roberta as model type.
        model_type = model.config.model_type
        model.config.model_type = "xlm-roberta"
        adapters.init(model)
        # reset model type (used later)
        model.config.model_type = model_type
        if not args.lora_path:
            for file in [
                "adapter_config.json",
                "head_config.json",
                "pytorch_adapter.bin",
                "pytorch_model_head.bin",
            ]:
                hf_hub_download(
                    repo_id=args.model_name_or_path,
                    subfolder=f"loras/{args.style_or_domain}/{args.language}/{args.use_lora}",
                    filename=file,
                    local_dir=Constants.CACHE_DIR,
                )
            lora_load_path = str(Constants.CACHE_DIR / "loras" / args.style_or_domain / args.language)
        else:
            lora_load_path = args.lora_path

        print(f"Using LoRA weights from {lora_load_path}.")
        model.load_adapter(
            lora_load_path,
            set_active=True,
            with_head=True,
            load_as="sat-lora",
        )
        # merge lora weights into transformer for 0 efficiency overhead
        model.merge_adapter("sat-lora")
        print("LoRA setup done.")
    # LoRA setup done, model is now ready for export.

    # Convert model to FP16 or FP32
    if args.device == 'cuda' and args.use_fp16:
        model = model.half()
        precision = "FP16"
    else:
        precision = "FP32"  # Default to FP32

    # Create output model name
    output_model_name = f"model.onnx"
    
    # Export model
    torch.onnx.export(
        model,
        {
            "input_ids": torch.randint(0, model.config.vocab_size, (1, 1), dtype=torch.int64, device=args.device),
            "attention_mask": torch.randn((1, 1), dtype=torch.float16, device=args.device),
        },
        output_dir / output_model_name,
        verbose=True,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch", 1: "sequence"},
        },
    )

    # Optimize model
    m = optimize_model(
        str(output_dir / output_model_name),
        model_type="bert",
        num_heads=0,
        hidden_size=0,
        optimization_options=None,
        opt_level=2, # from 0 to 3
        use_gpu=False,
    )

    optimized_model_path = output_dir / f"model_optimized.onnx"
    onnx.save_model(m.model, optimized_model_path)

    # Checking model
    onnx_model = onnx.load(output_dir / output_model_name)
    print(onnx.checker.check_model(onnx_model, full_check=True))

    # Upload mô hình lên Hugging Face Hub
    if args.upload_to_hub:
        api = HfApi()

        api.upload_file(
            path_or_fileobj=output_dir / f"model_optimized_{precision}.onnx",
            path_in_repo="model_optimized.onnx",
            repo_id=args.model_name_or_path,
        )
        api.upload_file(
            path_or_fileobj=output_dir / f"model_{precision}.onnx",
            path_in_repo="model.onnx",
            repo_id=args.model_name_or_path,
        )
