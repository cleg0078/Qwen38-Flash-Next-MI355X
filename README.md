# Qwen3.8-Flash-Next on AMD MI355X

## Running Qwen3.8-Flash-Next on a Single AMD MI355X

I wanted to test a **very high-end AMD GPU** and compare it with my **2-node NVIDIA DGX Spark cluster**.

The target was simple:

> Run **Qwen3.8-Flash-Next** on a **single AMD MI355X** with **FP8 + MTP enabled**.

It was definitely **not working out of the box**, but I finally got it running.

---

## Hardware

I rented an **AMD MI355X GPU Droplet from DigitalOcean** for approximately:

**$4.50/hour**

At that price, the GPU needs to be put to good use. :-)

The configuration tested here uses:

- **1× AMD MI355X**
- **Qwen3.8-Flash-Next FP8**
- **vLLM**
- **ROCm**
- **MTP enabled**
- Up to **300K context** depending on configuration

---

## Performance

With the FP8 model and MTP enabled, I reached approximately:

**180–270 tokens/s with a single stream**

Performance varies depending on:

- prompt length
- context size
- MTP acceptance rate
- cache usage
- workload
- vLLM configuration

I plan to test this setup more extensively with real-world agent workloads.

---

## MXFP4 Attempt

I also tried to run the **MXFP4 version**.

That turned out to require considerably more work.

After approximately **4 hours of testing**, I stopped for the moment.

I managed to load the model only after disabling most of the GPU hardware acceleration paths, but performance was poor:

**~20 tokens/s**

So although the model could load, this configuration was not useful.

I may add an MXFP4 script later after doing more investigation.

---

# Available Scripts

There are currently two scripts.

### 1. First functional version

```bash
qwen38-mi355x-fp8-final-functional-all-in-one.sh
```

This was my first working configuration.

---

### 2. More stable version for longer workloads

```bash
qwen38-mi355x-fp8-v8-no-prefix-cache-diagnostic.sh
```

This version can run longer tasks.

**Prefix caching is disabled** because I encountered GPU core dumps during longer workloads when prefix caching was enabled.

This is currently the version I recommend for testing.

---

# Environment Variables

Before running the script, configure the following environment variables:

```bash
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE=0

export MTP_TOKENS=3
export MAX_NUM_SEQS=4
export GPU_MEMORY_UTILIZATION=0.90
```

You also need a Hugging Face token to download the model:

```bash
export HF_TOKEN="your_huggingface_token"
```

And an API key to protect the vLLM endpoint:

```bash
export VLLM_API_KEY="$(openssl rand -hex 32)"
```

You can also provide your own API key:

```bash
export VLLM_API_KEY="your_api_key"
```

---

# Installation

Run:

```bash
./qwen38-mi355x-fp8-v8-no-prefix-cache-diagnostic.sh install
```

Then build the required environment:

```bash
./qwen38-mi355x-fp8-v8-no-prefix-cache-diagnostic.sh build
```

Check that everything is correctly configured:

```bash
./qwen38-mi355x-fp8-v8-no-prefix-cache-diagnostic.sh check
```

Finally, start the server:

```bash
./qwen38-mi355x-fp8-v8-no-prefix-cache-diagnostic.sh start
```

---

# Recommended Configuration

The configuration that currently works best for me is:

```bash
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE=0
export MTP_TOKENS=3
export MAX_NUM_SEQS=4
export GPU_MEMORY_UTILIZATION=0.90
```

with:

- **Qwen3.8-Flash-Next FP8**
- **1× AMD MI355X**
- **MTP = 3**
- **AITER enabled**
- **AITER MoE disabled**
- **Prefix cache disabled for stability**

---

## Notes

This is probably **not the optimal MI355X configuration**.

It is simply a configuration that **works**, which was the initial objective.

There is certainly room for further tuning of:

- AITER
- MoE kernels
- prefix caching
- MTP
- batch size
- concurrency
- context length
- ROCm kernel selection
- memory utilization

Contributions, tests, and better configurations are welcome.

**Have fun!**