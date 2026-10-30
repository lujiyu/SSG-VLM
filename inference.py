import os
import json
import gc
import time
import re
import argparse
from pathlib import Path
import torch
from PIL import Image
from datasets import load_dataset
from unsloth import FastVisionModel


# ============================================================
# 0. Parameters
# ============================================================


parser = argparse.ArgumentParser()

parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="checkpoint，eg. checkpoint-141 / checkpoint-282 / checkpoint-423 / final"
)

args = parser.parse_args()

MODEL_NAME = args.checkpoint


# ============================================================
# GPU
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")


DEVICE = torch.device("cuda:0")

torch.cuda.set_device(0)

GPU_ID = torch.cuda.current_device()


print("=" * 70, flush=True)

print(
    f"Visible GPU count : {torch.cuda.device_count()}",
    flush=True
)

print(
    f"Current device    : cuda:{GPU_ID}",
    flush=True
)

print(
    f"Device name       : "
    f"{torch.cuda.get_device_name(0)}",
    flush=True
)

print(
    f"Checkpoint        : {MODEL_NAME}",
    flush=True
)

print("=" * 70, flush=True)


# ============================================================
# 3. Path
# ============================================================

MODEL_PATH = (
    "./models/Qwen3.5-4B"
)

LORA_ROOT = Path(
    "./outputs/"
    "qwen3.5-4b-safety-lora-unsloth_ration0922c75b"
)

DATA_PATH = (
    "./data/sample_50000_ratio/test_5.0.jsonl"
)

OUTPUT_ROOT = LORA_ROOT


# ============================================================
# 4. checkpoint
# ============================================================

CHECKPOINTS = sorted(
    LORA_ROOT.glob("checkpoint-*"),
    key=lambda x: int(
        x.name.split("-")[-1]
    )
)

AVAILABLE_MODELS = [
    checkpoint.name
    for checkpoint in CHECKPOINTS
]

AVAILABLE_MODELS.append(
    "final"
)

print(
    "\nAvailable checkpoints:",
    flush=True
)

for name in AVAILABLE_MODELS:

    print(
        f"  - {name}",
        flush=True
    )


if MODEL_NAME not in AVAILABLE_MODELS:

    raise RuntimeError(
        f"\nInvalid checkpoint: "
        f"{MODEL_NAME}\n"
        f"Available: "
        f"{AVAILABLE_MODELS}"
    )


# ------------------------------------------------------------
# checkpoint path
# ------------------------------------------------------------

if MODEL_NAME == "final":

    MODEL_LORA_PATH = LORA_ROOT

else:

    MODEL_LORA_PATH = (
        LORA_ROOT / MODEL_NAME
    )


print(
    f"\nModel path: "
    f"{MODEL_LORA_PATH}",
    flush=True
)


if not MODEL_LORA_PATH.exists():

    raise RuntimeError(
        f"Model path does not exist:\n"
        f"{MODEL_LORA_PATH}"
    )


# ============================================================
# 5. Prompt
# ============================================================

PROMPT = """
You are a construction site safety hazard detection expert.

Please conduct a comprehensive inspection of the entire construction site image and identify all safety risks that can be directly observed from the image.

Please use the following inspection approach:

First, start from the overall scene and scan the personnel, equipment, work areas, edges, passageways, materials, electrical facilities, and other construction facilities in the image.

Then check each of these objects one by one for safety risks defined in the training data.

Identify and output different risks in the same image separately; do not stop checking just because one risk has already been found.

For multiple risks that can be clearly determined visually, output them separately even if these risks belong to the same risk category.

For risks with clear or strong visual evidence in the image, identify and output them as much as possible; do not easily omit them because the risk is not obvious, the target is small, or other risks exist at the same time.

Only output when there is direct visual evidence in the image; do not make inferences based on information outside the image.

Special attention:

Do not output only the most obvious risks.

Do not ignore other possible risks in an area just because one type of risk has already been identified there.

Do not actively merge different risks in order to reduce the number of outputs.

Scan the entire image completely before generating the final result.

Before outputting the result, check the entire image again to confirm whether there are other visible risks that have been missed.

For risks with clear visual evidence, identify them even if their targets are small or they are not in the most prominent area of the image.

risk_category and des must strictly use the standard labels in the training data.

bbox must accurately cover the direct visual evidence area corresponding to the risk, in the format [x1,y1,x2,y2].
The framing range should closely correspond to the visual evidence of the risk, avoid obviously oversized bounding boxes, and must not include large areas unrelated to the risk.

Finally, output only JSON, and do not output the analysis process, explanations, or other text.

Output format:
{
"risk": true,
"risks": [
{
"risk_category": "...",
"des": "...",
"bbox": [x1,y1,x2,y2]
}
]
}
"""

# ============================================================
# 6. Load dataset
# ============================================================

print(
    "\nLoading dataset...",
    flush=True
)

dataset = load_dataset(
    "json",
    data_files=DATA_PATH,
    split="train",
)

TOTAL_SAMPLES = len(dataset)

print(
    f"Total samples: "
    f"{TOTAL_SAMPLES}",
    flush=True
)

print(
    "This process will infer ALL samples.",
    flush=True
)


# ============================================================
# 7. Generate function
# ============================================================

def generate_response(
    model,
    processor,
    image_path,
    prompt,
    global_idx=None,
):

    # --------------------------------------------------------
    # OPEN IMAGE
    # --------------------------------------------------------

    print(
        f"[GPU {GPU_ID}] "
        f"Global {global_idx} | "
        f"OPEN_IMAGE | "
        f"{image_path}",
        flush=True
    )

    image = Image.open(
        image_path
    ).convert("RGB")

    print(
        f"[GPU {GPU_ID}] "
        f"Global {global_idx} | "
        f"IMAGE_SIZE | "
        f"{image.size}",
        flush=True
    )

    # --------------------------------------------------------
    # Chat template
    # --------------------------------------------------------

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]

    print(
        f"[GPU {GPU_ID}] "
        f"Global {global_idx} | "
        f"CHAT_TEMPLATE START",
        flush=True
    )

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    # --------------------------------------------------------
    # Processor
    # --------------------------------------------------------

    print(
        f"[GPU {GPU_ID}] "
        f"Global {global_idx} | "
        f"PROCESSOR START",
        flush=True
    )

    inputs = processor(
        text=text,
        images=[image],
        return_tensors="pt",
        truncation=False,
    )

    inputs = {
        k: v.to(DEVICE)
        if hasattr(v, "to")
        else v
        for k, v in inputs.items()
    }

    if "input_ids" in inputs:

        print(
            f"[GPU {GPU_ID}] "
            f"Global {global_idx} | "
            f"INPUT_TOKENS | "
            f"{inputs['input_ids'].shape[1]}",
            flush=True
        )

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    print(
        f"[GPU {GPU_ID}] "
        f"Global {global_idx} | "
        f"GENERATE START",
        flush=True
    )

    t_generate = time.time()

    with torch.inference_mode():

        generated_ids = model.generate(
            **inputs,

            max_new_tokens=1024,

            do_sample=False,

            repetition_penalty=1.05,
        )

    generate_time = (
        time.time()
        - t_generate
    )

    print(
        f"[GPU {GPU_ID}] "
        f"Global {global_idx} | "
        f"GENERATE DONE | "
        f"{generate_time:.2f}s",
        flush=True
    )

    # --------------------------------------------------------
    # Remove input tokens
    # --------------------------------------------------------

    input_length = inputs[
        "input_ids"
    ].shape[1]

    generated_ids = generated_ids[
        :,
        input_length:
    ]

    # --------------------------------------------------------
    # Decode
    # --------------------------------------------------------

    response = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return response


# ============================================================
# 8. Load model
# ============================================================

print(
    "\n" + "=" * 70,
    flush=True
)

print(
    f"START MODEL: "
    f"{MODEL_NAME}",
    flush=True
)

print(
    f"Loading: "
    f"{MODEL_LORA_PATH}",
    flush=True
)

model_load_start = time.time()


model, processor = (
    FastVisionModel.from_pretrained(
        model_name=str(
            MODEL_LORA_PATH
        ),

        # ----------------------------------------------------
        # 4bit inference
        # ----------------------------------------------------

        load_in_4bit=True,
        load_in_16bit=False,

        full_finetuning=False,
    )
)


FastVisionModel.for_inference(
    model
)

model.eval()


model_load_time = (
    time.time()
    - model_load_start
)


print(
    f"[GPU {GPU_ID}] "
    f"Model ready: "
    f"{MODEL_NAME}",
    flush=True
)

print(
    f"[GPU {GPU_ID}] "
    f"Model loading time: "
    f"{model_load_time:.2f}s",
    flush=True
)


# ============================================================
# 9. Output
# ============================================================

checkpoint_output_dir = (
    OUTPUT_ROOT / MODEL_NAME
)

checkpoint_output_dir.mkdir(
    parents=True,
    exist_ok=True,
)


# ------------------------------------------------------------
# each GPU
# ------------------------------------------------------------

local_output_path = (
    checkpoint_output_dir /
    f"predictions_gpu{GPU_ID}.jsonl"
)


print(
    f"[GPU {GPU_ID}] "
    f"Output: "
    f"{local_output_path}",
    flush=True
)


# ============================================================
# 10. Full inference
# ============================================================

checkpoint_start_time = time.time()


with open(
    local_output_path,
    "w",
    encoding="utf-8",
) as f:

    for global_idx, sample in enumerate(
        dataset
    ):

        print(
            "\n" + "-" * 70,
            flush=True
        )

        print(
            f"[GPU {GPU_ID}] "
            f"START "
            f"[{global_idx + 1}/"
            f"{TOTAL_SAMPLES}]",
            flush=True
        )

        image_path = (
            sample["messages"][0]
            ["content"][0]["image"]
        )

        ground_truth = (
            sample["messages"][1]
            ["content"][0]["text"]
        )

        sample_start_time = time.time()

        try:

            # ------------------------------------------------
            # Generate
            # ------------------------------------------------

            prediction = generate_response(
                model=model,
                processor=processor,
                image_path=image_path,
                prompt=PROMPT,
                global_idx=global_idx,
            )

            # ------------------------------------------------
            # Extract JSON
            # ------------------------------------------------

            #re_prediction = extract_json(prediction)

            # ------------------------------------------------
            # Validate
            # ------------------------------------------------

            #re_prediction = (validate_prediction(re_prediction))

            # ------------------------------------------------
            # Result
            # ------------------------------------------------

            result = {
                "index": global_idx,
                "image": image_path,
                "prediction": prediction,
                #"re_truth": re_prediction,
                "ground_truth": ground_truth,
            }

            sample_time = (
                time.time()
                - sample_start_time
            )

            print(
                f"[GPU {GPU_ID}] "
                f"DONE "
                f"Global index: "
                f"{global_idx} | "
                f"time={sample_time:.2f}s",
                flush=True,
            )

            print(
                f"Real Prediction: "
                f"{prediction}"
            )

        except Exception as e:

            sample_time = (
                time.time()
                - sample_start_time
            )

            print(
                f"[GPU {GPU_ID}] "
                f"ERROR "
                f"Global index="
                f"{global_idx} | "
                f"time={sample_time:.2f}s | "
                f"{repr(e)}",
                flush=True,
            )

            print(
                f"Real Prediction: "
                f"{prediction if 'prediction' in locals() else None}",
                flush=True
            )

            result = {
                "index": global_idx,
                "image": image_path,
                "prediction": None,
                #"re_truth": None,
                "ground_truth": ground_truth,
                "error": repr(e),
            }

        # ----------------------------------------------------
        # Write immediately
        # ----------------------------------------------------

        f.write(
            json.dumps(
                result,
                ensure_ascii=False,
            ) + "\n"
        )

        f.flush()


# ============================================================
# 11. Finished
# ============================================================

checkpoint_time = (
    time.time()
    - checkpoint_start_time
)

print(
    "\n" + "=" * 70,
    flush=True
)

print(
    f"[GPU {GPU_ID}] "
    f"FINISHED INFERENCE",
    flush=True
)

print(
    f"[GPU {GPU_ID}] "
    f"Checkpoint: "
    f"{MODEL_NAME}",
    flush=True
)

print(
    f"[GPU {GPU_ID}] "
    f"Total samples: "
    f"{TOTAL_SAMPLES}",
    flush=True
)

print(
    f"[GPU {GPU_ID}] "
    f"Inference time: "
    f"{checkpoint_time:.2f}s",
    flush=True
)

print(
    f"[GPU {GPU_ID}] "
    f"Saved: "
    f"{local_output_path}",
    flush=True
)

print(
    "=" * 70,
    flush=True
)


# ============================================================
# 12. Release
# ============================================================

del model
del processor

gc.collect()

torch.cuda.empty_cache()

print(
    f"[GPU {GPU_ID}] "
    f"Model released.",
    flush=True
)
