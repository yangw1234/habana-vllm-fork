bs=64
in_len=1024
out_len=1024
total_len=$((in_len + out_len))
model="meta-llama/Meta-Llama-3.1-8B-Instruct"

VLLM_SKIP_WARMUP=true python benchmarks/benchmark_throughput.py \
    --model ${model} \
    --device hpu \
    --backend vllm \
    --num-prompts ${bs} \
    --input_len ${in_len} \
    --output_len ${out_len} \
    --max_model_len ${total_len} \
    --dtype bfloat16 \
    --gpu-memory-util 0.9 \
    --use-v2-block-manager 2>&1 | tee benchmark_logs/offline_bf16_bs${bs}_i${in_len}_o${out_len}_fwd_lat.log