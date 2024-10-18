#VLLM_LOGGING_LEVEL=WARN VLLM_SKIP_WARMUP=true python -m vllm.entrypoints.openai.api_server --port 8080 --model meta-llama/Meta-Llama-3.1-8B-Instruct --tensor-parallel-size 1 --max-num-seqs 128 --disable-log-requests --dtype bfloat16 --block-size 128 --gpu-memory-util 0.9 --num-lookahead-slots 1 --use-v2-block-manager  --max-model-len 4096 | tee vllm_serving_llama3.1-8B-Instruct.log
bs=128
in_len=1024
out_len=1024
model="meta-llama/Meta-Llama-3.1-8B-Instruct"
echo "Start to warmup"
for i in 1 2;do
	python benchmarks/benchmark_serving.py --backend vllm --model $model --dataset-name sonnet --dataset-path benchmarks/sonnet.txt --request-rate 512 --num-prompts ${bs} --port 8080 --sonnet-input-len ${in_len} --sonnet-output-len ${out_len} --sonnet-prefix-len 100
done

echo "Start to benchmark"
python benchmarks/benchmark_serving.py --backend vllm --model ${model} --dataset-name sonnet --dataset-path benchmarks/sonnet.txt --request-rate 512 --num-prompts ${bs} --port 8080 --sonnet-input-len ${in_len} --sonnet-output-len ${out_len} --sonnet-prefix-len 100 | tee benchmark_logs/benchmark_serving_Llama-3.1-8B-Instruct_sonnet_bs${bs}_i${in_len}_o${out_len}_0attn.txt

kill 1525
