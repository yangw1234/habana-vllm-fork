VLLM_SKIP_WARMUP=true python benchmarks/benchmark_throughput.py \
    --model "meta-llama/Llama-3.2-1B" --device hpu \
    --backend vllm --num-prompts 32 --input_len 1024 --output_len 1024 \
    --dtype bfloat16 --gpu-memory-util 0.9 \
    --use-v2-block-manager \
    --max-model-len 4096