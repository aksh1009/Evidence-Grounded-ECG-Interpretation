"""Step 6b - report generation: template baseline, zero-shot LLM, LoRA fine-tuned LLM.

Fine-tuning details that the old notebook got wrong are handled here:
* prompt and answer are ONE sequence (chat template); loss only on answer tokens (prompt = -100);
* padding tokens are masked out of the loss;
* LoRA is applied once, on all attention + MLP projections.
"""
from __future__ import annotations

import math
import random
import time

import torch

from .claims_builder import CLASS_NAMES

SYSTEM = ("You are an ECG reporting assistant. Write a concise clinical 12-lead ECG report in the style of a "
          "cardiologist. Use ONLY the evidence provided: measurement findings [M#] and classifier findings [C#]. "
          "Never mention a finding that is not in the evidence. Classifier findings that are contradicted by "
          "measurements or have low probability should be omitted or stated with uncertainty.")


def build_messages(evidence_text: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Evidence:\n{evidence_text}\n\nWrite the ECG report."}]


# --------------------------------------------------------------------------- template baseline
RHYTHM = ["AFIB", "AFLT", "PACE", "SVTAC", "PSVT", "STACH", "SBRAD", "SARRH", "SR", "BIGU", "TRIGU", "SVARR"]


def template_report(ev: dict) -> str:
    dx = {c["concept"]: c for c in ev["diagnoses"] if not c["contradicted_by"]}
    meas = {c["concept"]: c for c in ev["findings"]}
    parts = []
    rh = [r for r in RHYTHM if r in dx]
    if rh:
        parts.append(" with ".join(CLASS_NAMES[r] for r in rh[:2]))
    elif "IRREG" in meas and "NO_P" in meas:
        parts.append("irregular rhythm without consistent p waves")
    if "HR" in meas:
        parts.append(f"heart rate {meas['HR']['value']:.0f} bpm")
    for k in ("LAD", "RAD", "EXT_AX", "LONG_PR", "LONG_QT", "WIDE_QRS", "LOWV"):
        if k in meas:
            parts.append(f"{meas[k]['finding']} ({meas[k]['value']:.0f} {meas[k]['unit']})"
                         if meas[k]["unit"] in ("ms", "deg") else meas[k]["finding"])
    for k, c in dx.items():
        if k in RHYTHM or k == "NORM":
            continue
        parts.append(c["finding"] if c["probability"] >= 0.5 else f"possible {c['finding']}")
    if "NORM" in dx and len(parts) <= 2:
        parts.append("otherwise normal ecg")
    return ". ".join(parts) + "." if parts else "no definite pathology."


# --------------------------------------------------------------------------- model loading
def load_llm(base: str, adapter: str | None = None, device: str = "cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16).to(device)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return tok, model


@torch.no_grad()
def generate(tok, model, evidence_texts: list[str], bs: int = 16, max_new_tokens: int = 160) -> list[str]:
    out = []
    for i in range(0, len(evidence_texts), bs):
        prompts = [tok.apply_chat_template(build_messages(t), tokenize=False, add_generation_prompt=True)
                   for t in evidence_texts[i:i + bs]]
        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id, repetition_penalty=1.05)
        out += [s.strip() for s in tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)]
    return out


# --------------------------------------------------------------------------- fine-tuning
def encode_example(tok, evidence_text: str, target: str, max_len: int) -> dict | None:
    prompt = tok.apply_chat_template(build_messages(evidence_text), tokenize=False, add_generation_prompt=True)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tok(target + tok.eos_token, add_special_tokens=False)["input_ids"]
    if len(p_ids) + len(a_ids) > max_len:
        return None
    return {"input_ids": p_ids + a_ids, "labels": [-100] * len(p_ids) + a_ids}


def _collate(batch, pad_id):
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), n), -100, dtype=torch.long)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        L = len(b["input_ids"])
        ids[i, :L] = torch.tensor(b["input_ids"])
        lab[i, :L] = torch.tensor(b["labels"])
        att[i, :L] = 1
    return ids, att, lab


def answer_only_loss(model, ids, att, lab):
    """Causal-LM loss computed only at answer positions.

    Equivalent to the standard loss with prompt labels = -100, but the 151k-word output layer is
    applied only where a label exists (~10% of positions), cutting logits memory ~10x.
    """
    base = model.get_base_model()
    h = base.model(input_ids=ids, attention_mask=att).last_hidden_state      # (B, T, H)
    tgt = lab[:, 1:]                                                          # next-token targets
    keep = tgt != -100
    logits = base.lm_head(h[:, :-1][keep]).float()                            # (N_answer, V)
    return torch.nn.functional.cross_entropy(logits, tgt[keep])


def finetune_lora(cfg, train_pairs, val_pairs, out_dir, log=print):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    c = cfg.llm
    tok = AutoTokenizer.from_pretrained(c.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(c.base_model, dtype=torch.bfloat16).to("cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=c.lora_r, lora_alpha=c.lora_alpha, lora_dropout=c.lora_dropout, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    ntr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"trainable params: {ntr/1e6:.1f}M")

    tr = [e for e in (encode_example(tok, a, b, c.max_len) for a, b in train_pairs) if e]
    va = [e for e in (encode_example(tok, a, b, c.max_len) for a, b in val_pairs) if e]
    log(f"encoded train {len(tr)}/{len(train_pairs)}  val {len(va)}/{len(val_pairs)}")
    random.Random(0).shuffle(tr)
    tr.sort(key=lambda e: len(e["input_ids"]) // 64 + random.random())   # light length bucketing
    batches = [tr[i:i + c.batch_size] for i in range(0, len(tr), c.batch_size)]
    random.Random(1).shuffle(batches)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=c.lr, weight_decay=0.0)
    total = math.ceil(len(batches) * c.epochs / c.grad_accum)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 50) * max(0.0, 0.5 * (1 + math.cos(math.pi * s / max(1, total)))))

    def val_loss():
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for i in range(0, min(len(va), 400), c.batch_size):
                ids, att, lab = (t.cuda() for t in _collate(va[i:i + c.batch_size], tok.pad_token_id))
                loss = answer_only_loss(model, ids, att, lab)
                tot += loss.item()
                n += 1
        model.train()
        return tot / max(1, n)

    log(f"val loss before training: {val_loss():.4f}")
    model.train()
    step, t0 = 0, time.time()
    for ep in range(c.epochs):
        for bi, b in enumerate(batches):
            ids, att, lab = (t.cuda() for t in _collate(b, tok.pad_token_id))
            loss = answer_only_loss(model, ids, att, lab) / c.grad_accum
            loss.backward()
            if (bi + 1) % c.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 50 == 0:
                    log(f"step {step}/{total} loss {loss.item()*c.grad_accum:.4f} ({time.time()-t0:.0f}s)")
                if step % 300 == 0:
                    log(f"step {step} val loss {val_loss():.4f}")
    log(f"final val loss {val_loss():.4f}")
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    return out_dir
