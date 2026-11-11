"""Build the fixed zero-shot classification heads from the pretrained CLIP text encoder.

For each task and class, every prompt template is embedded with the text
encoder, L2-normalised, averaged, and normalised again (the standard CLIP
zero-shot head, as used to fine-tune the task-specific encoders). The bundle also
stores the pretrained visual projection and the logit scale. Output:
``data/assets/zeroshot_heads.safetensors`` plus ``vision_config.json`` and
``param_shapes.json`` describing the image encoder.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = ROOT / "data" / "downloads" / "models" / "clip-vit-base-patch32"
OUT = ROOT / "data" / "assets"


@torch.no_grad()
def build(out_dir: Path = OUT) -> None:
    from safetensors.torch import save_file
    from transformers import CLIPConfig, CLIPModel, CLIPTokenizer, CLIPVisionModel

    model = CLIPModel(CLIPConfig.from_pretrained(str(BASE_DIR)))
    sd = torch.load(BASE_DIR / "pytorch_model.bin", map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.endswith("position_ids")]
    unexpected = [k for k in unexpected if not k.endswith("position_ids")]
    if missing or unexpected:
        raise SystemExit(f"pretrained checkpoint mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    model = model.float().eval()
    tok = CLIPTokenizer.from_pretrained(str(BASE_DIR))
    text = json.loads((ROOT / "data" / "zeroshot_text.json").read_text())
    manifest = json.loads((ROOT / "data" / "tasks.json").read_text())
    bundle: dict[str, torch.Tensor] = {}
    for task in manifest["tasks"]:
        name = task["name"]
        spec = text[name]
        rows = []
        for cls in spec["classnames"]:
            prompts = [t.format(c=cls) for t in spec["templates"]]
            enc = tok(prompts, padding=True, return_tensors="pt")
            out = model.text_model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            emb = model.text_projection(out.pooler_output)
            emb = F.normalize(emb, dim=-1).mean(dim=0)
            rows.append(F.normalize(emb, dim=-1))
        head = torch.stack(rows)
        assert head.shape[0] == task["num_classes"], name
        bundle[f"head.{name}"] = head.contiguous()
        print(name, tuple(head.shape))
    bundle["visual_projection"] = model.visual_projection.weight.detach().float().contiguous()
    bundle["logit_scale"] = model.logit_scale.exp().detach().float().reshape(1).contiguous()
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(bundle, str(out_dir / "zeroshot_heads.safetensors"))
    vcfg = model.config.vision_config.to_dict()
    (out_dir / "vision_config.json").write_text(json.dumps(vcfg, indent=1, sort_keys=True) + "\n")
    module = CLIPVisionModel(model.config.vision_config)
    shapes = {k: list(v.shape) for k, v in module.named_parameters()}
    (out_dir / "param_shapes.json").write_text(json.dumps(shapes, indent=1) + "\n")
    print("logit scale", float(bundle["logit_scale"]), "params", len(shapes))


if __name__ == "__main__":
    build()
