# Qwen38-Flash-Next-MI355X
How to run Qwen 3.8 Flash Next on a single MI355Xi?

I wanted to test a very high end GPU compared a 2 nodes DGX Spark cluster.

It was not out of the box. 

I rented a GPU droplet from DIGITAL OCEAN for 4.5$/hour. 

It's the FP8 model, MTP activated. 

I tried to run the mxfp4 model but needs a lot of work. I stopped after 4 hours. The script will be added maybe later. I need to prepare more. I had to disable all hardware gpu acceleration. It loads but only 20tok/s. 


For the FP8, I reached for 1 stream between 270 and 180 tokens/s. I will use it more for real projects. 4.5$/hour must be well used! 

Use this script as a template. I know it's probably not the best but it works... and it was the target. 

2 scripts now: 

- qwen38-mi355x-fp8-final-functional-all-in-one.sh = First try 
- qwen38-mi355x-fp8-v8-no-prefix-cache-diagnostic.sh = Can run longer task but prefill cache disabled to avoid GPU coredump! 

Both of them, you should set the following env vars:

export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE=0
export MTP_TOKENS=3
export MAX_NUM_SEQS=4
export GPU_MEMORY_UTILIZATION=0.90

export HF_TOKEN=your HF token to download the model
export VLLM_API_KEY=your generated token - you can use $(openssl rand -hex 32)

Then: 

qwen38-mi355x-fp8-v8-no-prefix-cache-diagnostic.sh install 
qwen38-mi355x-fp8-v8-no-prefix-cache-diagnostic.sh build 
qwen38-mi355x-fp8-v8-no-prefix-cache-diagnostic.sh check 
qwen38-mi355x-fp8-v8-no-prefix-cache-diagnostic.sh start

Have fun. 


