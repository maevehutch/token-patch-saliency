#!/usr/bin/env python3
"""
LLaVA Attention / Saliency Demo
================================

Refactored from a Jupyter notebook into a runnable CLI script.

This demo does three things, selectable via subcommands:

  1. `sensitivity` - loads one image + a question, runs the model, then
     re-runs it with increasing Gaussian noise added to the image, and
     plots how much the attention-derived saliency heatmap changes.

  2. `batch` - iterates over a JSON file of {image_index, question_string,
     type, answer} entries, runs the model on each image/question pair,
     builds an LLM attention map, overlays it on the image, and saves an
     SVG per question. This mirrors the big loop cell in the notebook.

  3. `metrics` - given saliency maps already saved as .npy files, runs the
     sparsity / minimality diagnostics from the notebook and prints a
     summary table.

IMPORTANT - external dependencies not included in the notebook itself:
  - The `llava` package (https://github.com/haotian-liu/LLaVA). Clone it
    and either `pip install -e .` inside it, or put it on PYTHONPATH /
    add it under `./models` (the notebook did `sys.path.append("./models")`).
  - A local `utils.py` providing: load_image, aggregate_llm_attention,
    aggregate_vit_attention, heterogenous_stack, show_mask_on_image.
    These were imported by the notebook but never defined in it, so
    they must already exist somewhere in your project - copy that file
    next to this script (or add its folder to PYTHONPATH).
  - A CUDA GPU is required in practice (the code loads the model in fp16
    and calls .to(device, dtype=torch.float16)); CPU will be extremely
    slow or may fail outright depending on your `llava` install.

See README.md for full setup instructions.
"""

import argparse
import json
import os
import shutil
import warnings

import numpy as np


# ---------------------------------------------------------------------------
# Lazy heavy imports
# ---------------------------------------------------------------------------
# torch / transformers / llava / cv2 are only imported inside the functions
# that need them, so that `python llava_attention_demo.py --help` and the
# `metrics` subcommand (which is pure numpy) work even if the heavy ML
# stack isn't installed yet.


# ============================================================================
# Model loading (from notebook cell "load_pretrained_model1")
# ============================================================================

def load_pretrained_model1(
    model_path,
    model_base,
    model_name,
    load_8bit=False,
    load_4bit=False,
    device_map="auto",
    device="cuda",
    use_flash_attn=False,
    **kwargs,
):
    """Load a LLaVA (or plain HF causal LM) checkpoint with eager attention
    forced on by default so that `output_attentions=True` works at generate
    time. Supports LoRA / mm-projector-only / full checkpoints."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
    from llava.model import (
        LlavaLlamaForCausalLM,
        LlavaMptForCausalLM,
        LlavaMistralForCausalLM,
    )
    from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs["device_map"] = {"": device}

    if load_8bit:
        kwargs["load_in_8bit"] = True
    elif load_4bit:
        kwargs["load_in_4bit"] = True
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        kwargs["torch_dtype"] = torch.float16

    if use_flash_attn:
        kwargs["attn_implementation"] = "flash_attention_2"
    else:
        # Force eager attention to enable attention output
        kwargs["attn_implementation"] = "eager"
    print(f"kwargs before loading model: {kwargs}")

    if "llava" in model_name.lower():
        if "lora" in model_name.lower() and model_base is None:
            warnings.warn(
                "There is `lora` in model name but no `model_base` is provided. "
                "If you are loading a LoRA model, please provide the `model_base` "
                "argument. Detailed instruction: "
                "https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged."
            )
        if "lora" in model_name.lower() and model_base is not None:
            from llava.model.language_model.llava_llama import LlavaConfig

            lora_cfg_pretrained = LlavaConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print("Loading LLaVA from base model...")
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs
            )
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(
                    torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype)
                )
                model.model.embed_tokens.weight = torch.nn.Parameter(
                    torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype)
                )

            print("Loading additional LLaVA weights...")
            if os.path.exists(os.path.join(model_path, "non_lora_trainables.bin")):
                non_lora_trainables = torch.load(
                    os.path.join(model_path, "non_lora_trainables.bin"), map_location="cpu"
                )
            else:
                from huggingface_hub import hf_hub_download

                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(repo_id=repo_id, filename=filename, subfolder=subfolder)
                    return torch.load(cache_file, map_location="cpu")

                non_lora_trainables = load_from_hf(model_path, "non_lora_trainables.bin")
            non_lora_trainables = {
                (k[11:] if k.startswith("base_model.") else k): v for k, v in non_lora_trainables.items()
            }
            if any(k.startswith("model.model.") for k in non_lora_trainables):
                non_lora_trainables = {
                    (k[6:] if k.startswith("model.") else k): v for k, v in non_lora_trainables.items()
                }
            model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel

            print("Loading LoRA weights...")
            model = PeftModel.from_pretrained(model, model_path)
            print("Merging LoRA weights...")
            model = model.merge_and_unload()
            print("Model is loaded...")
        elif model_base is not None:
            print("Loading LLaVA from base model...")
            if "mpt" in model_name.lower():
                if not os.path.isfile(os.path.join(model_path, "configuration_mpt.py")):
                    shutil.copyfile(
                        os.path.join(model_base, "configuration_mpt.py"),
                        os.path.join(model_path, "configuration_mpt.py"),
                    )
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
                cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                model = LlavaMptForCausalLM.from_pretrained(
                    model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                model = LlavaLlamaForCausalLM.from_pretrained(
                    model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs
                )

            mm_projector_weights = torch.load(os.path.join(model_path, "mm_projector.bin"), map_location="cpu")
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            if "mpt" in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = LlavaMptForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            elif "mistral" in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = LlavaMistralForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    else:
        if model_base is not None:
            from peft import PeftModel

            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print("Merging weights")
            model = model.merge_and_unload()
            print("Convert to FP16...")
            model.to(torch.float16)
        else:
            if "mpt" in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(
                    model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    image_processor = None

    if "llava" in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model(device_map=device_map)
        if device_map != "auto":
            vision_tower.to(device=device_map, dtype=torch.float16)
            if hasattr(vision_tower, "vision_tower") and hasattr(vision_tower.vision_tower, "config"):
                vision_tower.vision_tower.config._attn_implementation = "eager"
        image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len


def get_conv_mode(model_name):
    name = model_name.lower()
    if "llama-2" in name:
        return "llava_llama_2"
    elif "mistral" in name:
        return "mistral_instruct"
    elif "v1.6-34b" in name:
        return "chatml_direct"
    elif "v1" in name:
        return "llava_v1"
    elif "mpt" in name:
        return "mpt"
    return "llava_v0"


# ============================================================================
# Input-sensitivity test (from notebook cell "aa1ad5ab")
# ============================================================================

def add_noise_to_image(image, noise_level=0.1):
    """Add Gaussian noise to a PIL image."""
    from PIL import Image

    img_array = np.array(image, dtype=np.float32)
    noise = np.random.normal(0, noise_level * 255, img_array.shape)
    noisy = np.clip(img_array + noise, 0, 255)
    return Image.fromarray(np.uint8(noisy))


def get_heatmap_from_attention(attention_outputs):
    """Extract a 2D heatmap from attention (simple version): last layer,
    averaged over heads and batch, min-max normalized."""
    attn = attention_outputs[-1][0, :, :, :].mean(dim=0)  # (seq_len, seq_len)
    heatmap = attn.detach().cpu().numpy()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return heatmap


def compute_heatmap_distance(heatmap1, heatmap2):
    """Cosine similarity between two flattened heatmaps."""
    h1 = heatmap1.flatten()
    h2 = heatmap2.flatten()
    h1_norm = h1 / (np.linalg.norm(h1) + 1e-8)
    h2_norm = h2 / (np.linalg.norm(h2) + 1e-8)
    return float(np.dot(h1_norm, h2_norm))


def test_input_sensitivity_simple(
    model, image_processor, tokenizer, image_path, prompt_text, noise_levels=(0.05, 0.10, 0.15, 0.20)
):
    """Add increasing noise to an image and measure how much the
    attention-derived saliency heatmap changes relative to the clean image."""
    import torch
    from PIL import Image
    from llava.constants import DEFAULT_IMAGE_TOKEN
    from llava.mm_utils import process_images

    print("\n" + "=" * 70)
    print("INPUT SENSITIVITY TEST")
    print("=" * 70)
    print(f"Testing on image: {image_path}")
    print(f"Question: {prompt_text}\n")

    original_image = Image.open(image_path).convert("RGB")

    results = {"noise_level": [], "heatmap_similarity": []}

    print("Processing original image...")
    with torch.inference_mode():
        image_tensor, images = process_images([original_image], image_processor, model.config)
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

        inp = DEFAULT_IMAGE_TOKEN + "\n" + prompt_text
        inputs = tokenizer(inp, return_tensors="pt").to(model.device)

        output_orig = model.generate(
            **inputs,
            images=image_tensor,
            do_sample=False,
            max_new_tokens=100,
            output_attentions=True,
            return_dict_in_generate=True,
        )

        heatmap_original = get_heatmap_from_attention(output_orig["attentions"])
        response_original = tokenizer.decode(output_orig["sequences"][0], skip_special_tokens=True)

    print(f"Original response: {response_original}\n")

    for noise_level in noise_levels:
        print(f"Testing noise level: {noise_level:.2f}...", end=" ")

        noisy_image = add_noise_to_image(original_image, noise_level)

        with torch.inference_mode():
            image_tensor_noisy, _ = process_images([noisy_image], image_processor, model.config)
            image_tensor_noisy = image_tensor_noisy.to(model.device, dtype=torch.float16)

            output_noisy = model.generate(
                **inputs,
                images=image_tensor_noisy,
                do_sample=False,
                max_new_tokens=100,
                output_attentions=True,
                return_dict_in_generate=True,
            )

            heatmap_noisy = get_heatmap_from_attention(output_noisy["attentions"])
            response_noisy = tokenizer.decode(output_noisy["sequences"][0], skip_special_tokens=True)

        similarity = compute_heatmap_distance(heatmap_original, heatmap_noisy)

        results["noise_level"].append(noise_level)
        results["heatmap_similarity"].append(similarity)

        same_response = "✓" if response_original == response_noisy else "✗"
        print(f"Similarity: {similarity:.3f} {same_response}")

    return results


def plot_input_sensitivity(results, output_path="input_sensitivity.png"):
    """Plot noise level vs heatmap similarity, with interpretation bands."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        results["noise_level"],
        results["heatmap_similarity"],
        "o-",
        linewidth=2,
        markersize=10,
        color="#2E86AB",
    )

    ax.set_xlabel("Noise Level (std dev as % of pixel range)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Heatmap Similarity to Original", fontsize=12, fontweight="bold")
    ax.set_title(
        "INPUT SENSITIVITY: How stable is the saliency map to input noise?",
        fontsize=13,
        fontweight="bold",
    )

    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.1])

    ax.axhspan(0.85, 1.0, alpha=0.1, color="green", label="Robust (good)")
    ax.axhspan(0.5, 0.85, alpha=0.1, color="yellow", label="Moderate")
    ax.axhspan(0.0, 0.5, alpha=0.1, color="red", label="Sensitive (risky)")

    ax.legend(loc="lower left", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n✓ Saved plot to: {output_path}")
    plt.close(fig)


def interpret_results(results):
    """Print a human-readable interpretation of the sensitivity results."""
    similarities = np.array(results["heatmap_similarity"])
    avg_similarity = similarities.mean()

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    print(f"\nAverage similarity across noise levels: {avg_similarity:.3f}")

    if avg_similarity > 0.85:
        print("\n✓ ROBUST: Saliency is stable to input perturbations")
        print("  → Good for: High-stakes decisions (medical, legal, etc.)")
        print("  → Caution: May be missing some sensitivity to real changes")
    elif avg_similarity > 0.60:
        print("\n△ MODERATE: Saliency changes somewhat with input noise")
        print("  → This is normal for many real applications")
        print("  → Monitor: Check if changes correlate with output changes")
    else:
        print("\n✗ SENSITIVE: Saliency is highly affected by input noise")
        print("  → Caution: May not be trustworthy for interpretation")
        print("  → Consider: Different saliency method or model regularization")

    print("\nWhat this means:")
    print("- HIGH similarity (>0.85):  Heatmap stays same when image is noisy")
    print("- MEDIUM similarity (0.5-0.85): Heatmap changes proportionally to noise")
    print("- LOW similarity (<0.5):   Heatmap drastically changes with small noise")

    trend = similarities[-1] - similarities[0]
    if trend < -0.2:
        print("\n⚠️  Trend: Similarity decreases with more noise")
        print("    → Saliency becomes unstable at higher noise levels")
    elif trend > 0.2:
        print("\n⚠️  Trend: Similarity INCREASES with more noise (unusual)")
    else:
        print("\n→ Trend: Stable across noise levels")


# ============================================================================
# Sparsity / minimality metrics (from notebook cells near the end)
# ============================================================================

def sparsity_entropy(saliency_map):
    """Lower = more sparse."""
    saliency_flat = saliency_map.flatten()
    saliency_flat = saliency_flat / (np.sum(saliency_flat) + 1e-8)
    saliency_flat = np.clip(saliency_flat, a_min=1e-10, a_max=None)
    entropy = -np.sum(saliency_flat * np.log(saliency_flat))
    return float(entropy)


def sparsity_maxmin(saliency_map):
    """Higher = more sparse."""
    saliency_flat = saliency_map.flatten()
    saliency_flat = np.clip(saliency_flat, a_min=1e-8, a_max=None)
    return float(np.max(saliency_flat) / (np.min(saliency_flat) + 1e-8))


def sparsity_concentration(saliency_map, threshold=80):
    """Higher = more sparse. threshold=80 means top 20% of pixels."""
    saliency_flat = saliency_map.flatten()
    saliency_flat = saliency_flat / (np.sum(saliency_flat) + 1e-8)
    threshold_val = np.percentile(saliency_flat, threshold)
    return float(np.sum(saliency_flat[saliency_flat >= threshold_val]) * 100)


def test_sparsity(saliency_maps_dict):
    print("\n" + "=" * 70)
    print("SPARSITY TEST")
    print("=" * 70)
    print("(Higher entropy = more diffuse | Higher max/min = more focused)\n")

    for name, smap in saliency_maps_dict.items():
        entropy = sparsity_entropy(smap)
        maxmin = sparsity_maxmin(smap)
        conc80 = sparsity_concentration(smap, 80)
        conc90 = sparsity_concentration(smap, 90)
        print(
            f"{name:20} | Entropy: {entropy:6.3f} | Max/Min: {maxmin:7.2f} "
            f"| Conc@80%: {conc80:6.1f}% | Conc@90%: {conc90:6.1f}%"
        )
    print("=" * 70)


def minimality_cardinality(saliency_map, threshold=0.5):
    """Lower = more minimal. What % of pixels exceed threshold?"""
    saliency_norm = saliency_map / (np.max(saliency_map) + 1e-8)
    num_important = np.sum(saliency_norm >= threshold)
    total = saliency_map.size
    return float(100 * num_important / total)


def minimality_mass_ratio(saliency_map, target_ratio=0.9):
    """Lower = more minimal. What % of pixels needed for target_ratio of mass?"""
    saliency_flat = saliency_map.flatten()
    total_mass = np.sum(saliency_flat)
    sorted_saliency = np.sort(saliency_flat)[::-1]
    cumsum = np.cumsum(sorted_saliency)
    cumsum_norm = cumsum / (total_mass + 1e-8)
    idx = np.argmax(cumsum_norm >= target_ratio)
    pixels_needed = idx + 1
    return float(100 * pixels_needed / len(saliency_flat))


def minimality_threshold_ratio(saliency_map):
    """Higher = more minimal. Distinction between important/unimportant."""
    saliency_flat = saliency_map.flatten()
    saliency_flat = saliency_flat / (np.sum(saliency_flat) + 1e-8)
    median_val = np.median(saliency_flat)
    mean_val = np.mean(saliency_flat)
    ratio = median_val / mean_val if mean_val > 0 else 0
    return float(ratio)


def test_minimality(saliency_maps_dict):
    print("\n" + "=" * 70)
    print("MINIMALITY TEST")
    print("=" * 70)
    print("(Lower = more minimal | More pixels = less minimal)\n")

    for name, smap in saliency_maps_dict.items():
        card50 = minimality_cardinality(smap, 0.5)
        card70 = minimality_cardinality(smap, 0.7)
        mass90 = minimality_mass_ratio(smap, 0.9)
        mass95 = minimality_mass_ratio(smap, 0.95)
        threshold = minimality_threshold_ratio(smap)
        print(
            f"{name:20} | Card@50%: {card50:6.1f}% | Card@70%: {card70:6.1f}% "
            f"| Mass@90%: {mass90:6.1f}% | Mass@95%: {mass95:6.1f}% | Thresh: {threshold:5.2f}"
        )
    print("=" * 70)


def test_both(saliency_maps_dict):
    test_sparsity(saliency_maps_dict)
    test_minimality(saliency_maps_dict)

    print("\n" + "=" * 70)
    print("QUICK ASSESSMENT")
    print("=" * 70)

    for name, smap in saliency_maps_dict.items():
        entropy = sparsity_entropy(smap)
        mass90 = minimality_mass_ratio(smap, 0.9)

        if entropy < 2.5 and mass90 < 25:
            quality = "⭐ IDEAL (Sparse + Minimal)"
        elif entropy < 3.5 and mass90 < 40:
            quality = "✓ GOOD"
        elif entropy > 3.5 or mass90 > 40:
            quality = "✗ POOR (Diffuse or Extensive)"
        else:
            quality = "△ MODERATE"

        print(f"{name:20} → {quality}")
    print("=" * 70)


# ============================================================================
# Batch attention visualization over a QA dataset (from the big loop cell)
# ============================================================================

def run_batch(args):
    import torch
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from llava.conversation import conv_templates
    from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
    from llava.utils import disable_torch_init
    from utils import load_image, aggregate_llm_attention, heterogenous_stack, show_mask_on_image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model1(
        args.model_path,
        None,
        model_name,
        load_8bit=False,
        load_4bit=False,
        device_map=device,
        device=device,
        use_flash_attn=False,
    )

    grid_size = model.get_vision_tower().num_patches_per_side
    os.makedirs(args.output_dir, exist_ok=True)

    conv_mode = get_conv_mode(model_name)

    with open(args.qapairs_path, "r") as f:
        qa_pairs = json.load(f)
    qa_pairs = qa_pairs[: args.num_questions]

    for idx, qa in enumerate(qa_pairs):
        image_index = qa["image_index"]
        prompt_text = qa["question_string"]
        chart_type = qa["type"]
        image_path = os.path.join(args.images_dir, f"{image_index}.png")

        print(f"\n{'=' * 80}")
        print(f"Processing {idx + 1}/{len(qa_pairs)}: {chart_type}")
        print(f"Question: {prompt_text}")
        print(f"Image: {image_path}")
        print("=" * 80)

        try:
            image = load_image(image_path)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            continue

        image_tensor, images = process_images([image], image_processor, model.config)
        image = images[0]
        image_size = image.size
        if isinstance(image_tensor, list):
            image_tensor = [img.to(model.device, dtype=torch.float16) for img in image_tensor]
        else:
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)

        conv = conv_templates[conv_mode].copy()

        if model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + prompt_text
        else:
            inp = DEFAULT_IMAGE_TOKEN + "\n" + prompt_text

        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        prompt = prompt.replace(
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions. ",
            "",
        )

        input_ids = (
            tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .to(model.device)
        )

        with torch.inference_mode():
            outputs = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image_size],
                do_sample=False,
                max_new_tokens=512,
                use_cache=True,
                return_dict_in_generate=True,
                output_attentions=True,
            )

        response = tokenizer.decode(outputs["sequences"][0], skip_special_tokens=True).strip()
        print(f"\nResponse: {response}\n")

        aggregated_prompt_attention = []
        for layer in outputs["attentions"][0]:
            layer_attns = layer.squeeze(0)
            attns_per_head = layer_attns.mean(dim=0)
            cur = attns_per_head[:-1].cpu().clone()
            cur[1:, 0] = 0.0
            cur[1:] = cur[1:] / cur[1:].sum(-1, keepdim=True)
            aggregated_prompt_attention.append(cur)
        aggregated_prompt_attention = torch.stack(aggregated_prompt_attention).mean(dim=0)

        llm_attn_matrix = heterogenous_stack(
            [torch.tensor([1])]
            + list(aggregated_prompt_attention)
            + list(map(aggregate_llm_attention, outputs["attentions"]))
        )

        num_vision_patches = model.get_vision_tower().num_patches
        vision_token_start = len(tokenizer(prompt.split("<image>")[0], return_tensors="pt")["input_ids"][0])
        vision_token_end = vision_token_start + num_vision_patches

        input_token_len = num_vision_patches + len(input_ids[0]) - 1
        num_generated_tokens = len(outputs["attentions"])
        total_attn_len = llm_attn_matrix.shape[0]

        full_sequence = outputs["sequences"][0]
        generated_tokens = full_sequence[-num_generated_tokens:] if num_generated_tokens > 0 else torch.tensor([])

        output_token_start = input_token_len
        output_token_end = total_attn_len
        output_token_len = num_generated_tokens

        if output_token_len <= 0:
            print(f"Warning: No new tokens generated for question {idx + 1}")
            continue

        num_image_per_row = 8
        image_ratio = image_size[0] / image_size[1]
        num_rows = output_token_len // num_image_per_row + (1 if output_token_len % num_image_per_row != 0 else 0)
        num_rows = max(num_rows, 1)

        fig, axes = plt.subplots(
            num_rows,
            num_image_per_row,
            figsize=(16, (16 / num_image_per_row) * image_ratio * num_rows),
            dpi=150,
        )
        if num_rows == 1:
            axes = axes.reshape(1, -1)

        plt.subplots_adjust(wspace=0.1, hspace=0.3, top=0.95, bottom=0.05)

        output_token_inds = list(range(output_token_start, output_token_end))
        saliency_maps_collected = {}

        for i, ax in enumerate(axes.flatten()):
            if i >= output_token_len:
                ax.axis("off")
                continue

            target_token_ind = output_token_inds[i]
            if target_token_ind >= llm_attn_matrix.shape[0]:
                print(f"Warning: token index {target_token_ind} out of bounds, skipping")
                ax.axis("off")
                continue

            attn_weights_over_vis_tokens = llm_attn_matrix[target_token_ind][vision_token_start:vision_token_end]

            if len(attn_weights_over_vis_tokens) == 0 or attn_weights_over_vis_tokens.sum() == 0:
                print(f"Warning: No attention to vision tokens for token {i}")
                ax.axis("off")
                continue

            attn_weights_over_vis_tokens = attn_weights_over_vis_tokens / attn_weights_over_vis_tokens.sum()

            attn_over_image = attn_weights_over_vis_tokens.reshape(grid_size, grid_size)
            attn_over_image = attn_over_image / attn_over_image.max()
            attn_over_image = F.interpolate(
                attn_over_image.unsqueeze(0).unsqueeze(0),
                size=image_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze()
            saliency_maps_collected[f"token_{i}"] = attn_over_image.cpu().numpy()

            np_img = np.array(image)[:, :, ::-1]
            img_with_attn, heatmap = show_mask_on_image(np_img, attn_over_image.cpu().numpy())
            ax.imshow(img_with_attn if args.overlay else heatmap)

            if i < len(generated_tokens):
                token_id = generated_tokens[i]
                token_text = tokenizer.decode([token_id], skip_special_tokens=False).strip()
                ax.set_title(
                    token_text if token_text else f"[ID:{token_id}]",
                    fontsize=9,
                    pad=3,
                    color="black",
                    backgroundcolor="white",
                    weight="bold",
                )
            ax.axis("off")

        output_filename = os.path.join(args.output_dir, f"{chart_type}_{image_index}.svg")
        plt.tight_layout()
        plt.savefig(output_filename, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved visualization to: {output_filename}")

        if args.run_saliency_metrics and saliency_maps_collected:
            test_both(saliency_maps_collected)

    print(f"\n{'=' * 80}")
    print(f"Completed processing {len(qa_pairs)} questions")
    print(f"Visualizations saved in: {args.output_dir}")
    print("=" * 80)


def run_sensitivity(args):
    import torch
    from llava.mm_utils import get_model_name_from_path
    from llava.utils import disable_torch_init

    device = "cuda" if torch.cuda.is_available() else "cpu"
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model1(
        args.model_path,
        None,
        model_name,
        load_8bit=False,
        load_4bit=False,
        device_map=device,
        device=device,
        use_flash_attn=False,
    )

    results = test_input_sensitivity_simple(
        model,
        image_processor,
        tokenizer,
        args.image,
        args.question,
        noise_levels=args.noise_levels,
    )
    plot_input_sensitivity(results, output_path=args.output)
    interpret_results(results)


def run_metrics(args):
    saliency_maps = {}
    for path in args.saliency_maps:
        name = os.path.splitext(os.path.basename(path))[0]
        saliency_maps[name] = np.load(path)

    if not saliency_maps:
        print("No .npy saliency maps provided.")
        return

    test_both(saliency_maps)


# ============================================================================
# CLI
# ============================================================================

def build_parser():
    parser = argparse.ArgumentParser(
        description="LLaVA attention / saliency demo (see module docstring for setup requirements).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_sens = sub.add_parser("sensitivity", help="Run the input-noise sensitivity test on one image.")
    p_sens.add_argument("--model-path", default="liuhaotian/llava-v1.5-7b")
    p_sens.add_argument("--image", required=True, help="Path to an image file.")
    p_sens.add_argument("--question", required=True, help="Question to ask about the image.")
    p_sens.add_argument("--noise-levels", type=float, nargs="+", default=[0.05, 0.10, 0.15, 0.20])
    p_sens.add_argument("--output", default="input_sensitivity.png")
    p_sens.set_defaults(func=run_sensitivity)

    p_batch = sub.add_parser("batch", help="Batch-run attention visualization over a QA JSON dataset.")
    p_batch.add_argument("--model-path", default="liuhaotian/llava-v1.5-7b")
    p_batch.add_argument("--qapairs-path", default="image_indices.json")
    p_batch.add_argument("--images-dir", default="pngval/png/png")
    p_batch.add_argument("--output-dir", default="lava_out_attn")
    p_batch.add_argument("--num-questions", type=int, default=50)
    p_batch.add_argument("--overlay", action="store_true", default=True, help="Overlay heatmap on image (default).")
    p_batch.add_argument(
        "--heatmap-only", dest="overlay", action="store_false", help="Show raw heatmap instead of overlay."
    )
    p_batch.add_argument(
        "--run-saliency-metrics",
        action="store_true",
        help="Also run sparsity/minimality diagnostics per question.",
    )
    p_batch.set_defaults(func=run_batch)

    p_metrics = sub.add_parser("metrics", help="Run sparsity/minimality diagnostics on saved .npy saliency maps.")
    p_metrics.add_argument("saliency_maps", nargs="+", help="Paths to .npy saliency map files.")
    p_metrics.set_defaults(func=run_metrics)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
