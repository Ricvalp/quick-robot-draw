# Quick, Robot, Draw! — Task Overview

This repository repackages Google’s **Quick, Draw!** sketches into a robotics-friendly benchmark focused on drawing control. Each sketch is normalized, tokenized, and bundled into K-shot “prompt + query” episodes so that a policy learns to imitate human strokes. Two complementary learning settings are supported:

---

## 1. Simple Imitation Learning (Single-Sketch Supervision)

**Goal:** Learn a policy that reproduces an entire sketch from scratch (no prompts) by conditioning on the observed stroke history.

- **Data pathway:** `dataset/loader.py` emits single-sketch episodes; `dataset/diffusion.py` (diffusion) or `dataset/lstm.py` (SketchRNN) collate them into padded tensors.
- **Policies:**
  - `diffusion_policy/` – DiT-based diffusion policy that denoises future stroke chunks (`train_quickdraw.py`).
  - `lstm/` – SketchRNN VAE with BiLSTM encoder + autoregressive decoder (`train_quickdraw.py`).
- **Learning signal:** Standard behaviour cloning.
  - Diffusion: predict noise over a horizon window of stroke deltas + pen state.
  - LSTM: maximize stroke likelihood under a 2D Gaussian mixture and categorical pen-state head with KL-regularized latent space.
- **Usage:** Ideal as an intermediate check that models capture sketch statistics before tackling in-context prompts.

---

## 2. In-Context Imitation Learning (K-Shot Prompting)

**Goal:** At test time, a policy receives **K example sketches (prompts)** from the same semantic family plus a query sketch that must be completed. The model must extract the “concept” directly from the prompts and imitate it for the query—no gradient updates, only conditioning.

- **Episode format:** `dataset/episode_builder.py` concatenates `[START, prompt₁, SEP, …, RESET, START, query, STOP]` with structural bits marking transitions. `QuickDrawEpisodes` streams these on the fly with deterministic seeds and optional augmentations.
- **Planned policies:** Future transformers or sequence models that ingest the full prompt/query token stream and generate the missing query strokes in-context (e.g., attention over prompts, masked modeling, or hybrid diffusion/LSTM decoders).
- **Learning signal:** Behaviour cloning on the query portion while the prompt acts as context tokens; supports few-shot generalization across thousands of sketch families.
- **Why it matters:** Mirrors “promptable” robotics/imitation setups where the robot must infer a task from demonstrations provided at inference time, without per-task fine-tuning.

---

## Takeaways

1. **Simple imitation learning** verifies that a policy can model raw sketch dynamics (diffusion or LSTM engines already included).
2. **In-context imitation learning** scales that ability to prompt-conditioned generalization: same dataset, richer supervision strategy.
3. The shared dataset + loaders let you iterate quickly between both regimes, making this repo a benchmark for prompt-driven imitation policies.
