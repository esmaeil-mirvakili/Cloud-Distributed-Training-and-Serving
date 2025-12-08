Decisions for AWS runs (initial pass)
- Training models for Dolly-15K SFT: meta-llama/Llama-2-7b-chat-hf (mainline) and meta-llama/Llama-2-13b-chat-hf (scale-up for multi-GPU); fallback to mistralai/Mistral-7B-Instruct-v0.2 if Llama-2 access is blocked.
- Serving/inference (llama.cpp GGUF), biasing for small/fast: TinyLlama-1.1B-Chat-v1.0-GGUF (very low-latency CPU) and Phi-2-GGUF (~2.7B, stronger quality, still fast); keep Llama-2-7B/13B-GGUF for comparison if quotas allow.
- Quantizations to sweep in llama.cpp: F16 (non-quant baseline), Q8_0, Q6_K, Q4_K_M; record memory footprint per level. Default serving configs: Q4_K_M on CPU, Q6_K on GPU unless a sweep is running.

Unknowns/TBD before execution
- Hugging Face token/acceptance for Llama 2 weights; confirm whether to lock in the Mistral 7B fallback.
- Confirm availability of TinyLlama-1.1B-Chat GGUF and Phi-2 GGUF artifacts (likely from TheBloke) and allowed licenses.
- AWS instance types/quotas for single-GPU baseline and 2/4/8-GPU scaling (e.g., g5.2xlarge for 1 GPU, p4d.24xlarge for 8 GPU); budget per run.
- Exact S3 locations for Dolly-15K shards, WildChat quality subset, and converted GGUF artifacts.
- Prompt set/request shape for k6 and WildChat evaluations (OpenAI-compatible vs native llama.cpp server mode).
- Cost model inputs (on-demand vs spot pricing, target region, currency).

1) Training performance experiments (single GPU vs distributed)

A. Baseline: single-GPU fine-tuning run (control)
	•	Run SFT baseline on 1 GPU (Dolly-15K) with meta-llama/Llama-2-7b-chat-hf + LoRA; stretch: meta-llama/Llama-2-13b-chat-hf if access/quotas allow
	•	Collect:
	•	Step time distribution (mean, p95)
	•	Throughput (samples/sec or tokens/sec)
	•	GPU memory peak, GPU utilization, CPU/RAM (Prometheus)
	•	Wall-clock time to finish fixed number of steps/epochs

B. Distributed training scaling: 2, 4, (8) GPUs (or whatever you can actually get)
	•	Run the same fixed training workload on 2, 4, and 8 gpus (7B mainline)
	•	Collect the same metrics as baseline (step time, throughput, utilization, wall-clock)
	•	Report:
	•	Speedup = baseline_step_time / distributed_step_time
	•	Scaling efficiency = speedup / N
	•	Cost per run (see section 4)  ￼
	•	Script: `python experiments/training/run_training_experiments.py --config experiments/training/training_config.example.yaml --output outputs/training_results.json`
	•	Per-experiment: applies k8s overlay (baseline / 2/4/8 GPU), waits for rollout, captures trainer pod logs tail

⸻

1) Inference performance experiments (llama.cpp, CPU vs GPU, quantization, model sizes)

 Script: `python experiments/inference/run_inference_experiments.py --config experiments/inference/configs/inference_models.example.yaml --output outputs/inference_perf.json`

Your plan: compare quantized vs non-quantized, CPU vs GPU, and favor small/fast models (TinyLlama-1.1B-Chat-GGUF, Phi-2-GGUF) with optional Llama-2-7B/13B-GGUF comparison if accessible.  ￼￼

A. Quantization sweep (inference)
	•	For each model size you can deploy TinyLlama-1.1B-Chat-GGUF and Phi-2-GGUF on CPU:  ￼
	•	Run F16 (non-quant baseline)
	•	Run Q8_0, Q6_K, Q4_K_M (llama.cpp GGUF quant levels)
	•	For each: collect latency percentiles, tokens/sec, memory footprint, and cost per 1k tokens  ￼

B. CPU vs GPU comparison
	•	Repeat A on:
	•	GPU-backed nodes
	•	Deliverable: identify regimes where GPU wins on cost-performance and raw speed  ￼

⸻

1) Output quality experiments (WildChat references with similarity metrics)

Your plan: treat WildChat’s original ChatGPT responses as a reference and score ROUGE/BLEU/BERTScore.  ￼

A. Quality experiment
	•	Run inference on the fixed WildChat quality subset (100) on the TinyLlama-1.1B-Chat-GGUF and Phi-2-GGUF on CPU
	•	Script: `python experiments/quality/run_quality_experiments.py --config experiments/quality/quality_config.example.yaml --output outputs/quality_results.json`
	•	Input format: JSONL with fields `prompt` and `reference` (configurable)
	•	Compute:
	•	ROUGE, BLEU, BERTScore  ￼

⸻

4) Elastic serving + autoscaling experiments (HPA under k6 load profiles)

Your plan: HPA scaling behind NGINX, evaluated using steady/ramp, spike/stress, soak; measure SLO compliance and scaling time.  ￼

A. SLO compliance under load
	•	Script: `python experiments/autoscaling/run_autoscaling_experiments.py --config experiments/autoscaling/autoscaling_config.example.yaml --output outputs/autoscaling_results.json`
	•	Load: k6 script(s) per experiment (configurable); service is port-forwarded locally
	•	Measure and report:
	•	% requests meeting ttft P95 < 2s / 5s / 10s / 20s / 30s (from Prometheus)
	•	Latency P95, TTFT P95, latency bucket ratios (<1s, <5s, <10s)
	•	Resource efficiency: average CPU/GPU utilization, GPU memory (from Prometheus)
