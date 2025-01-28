"""
Example script demonstrating inference on Habana Gaudi processors using vLLM's HPU worker.
"""

import os
import torch
import habana_frameworks.torch as htorch
from vllm.config import (
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    DeviceConfig,
    CacheConfig,
    VllmConfig,
    ConfigFormat
)
from vllm.worker.hpu_worker import HPUWorker
from vllm.sampling_params import SamplingParams
from vllm.sequence import SequenceGroupMetadata, SequenceData
from array import array
from typing import Optional, Dict, Any, List
from vllm.engine.arg_utils import EngineArgs, TaskOption
from vllm.utils import get_distributed_init_method, get_ip, get_open_port
from vllm.worker.hpu_model_runner import ModelInputForHPUWithSamplingMetadata
from vllm.attention.backends.hpu_attn import HPUAttentionMetadata
from vllm.model_executor.sampling_metadata import SamplingMetadata, SequenceGroupToSample
from vllm.worker.hpu_worker import WorkerInput
from vllm.model_executor.sampling_metadata import SamplingType


from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

class HPUTPWorker():


    def __init__(self, model: str):

        self.vllm_engine_args = create_engine_args(model=model, device="hpu", dtype="bfloat16")
        self.vllm_config = self.vllm_engine_args.create_engine_config()
        self.distributed_init_method = get_distributed_init_method(
            get_ip(), get_open_port())

        # Initialize HPU worker
        self.vllm_worker = HPUWorker(
            vllm_config=self.vllm_config,
            local_rank=0,
            rank=0,
            distributed_init_method=self.distributed_init_method,
            is_driver_worker=True
        )
        self.vllm_worker.init_device()
        self.vllm_worker.load_model()
        self.tp_rank = self.vllm_worker.local_rank
        self.device = self.vllm_worker.device

        self.req_to_token_pool = ReqToTokenPool(
            size=self.vllm_config.max_num_tokens,
            max_context_len=self.vllm_config.context_len,
            device=self.device,
            enable_memory_saver=False
        )
        


def create_engine_args(
    model: str,
    tokenizer: Optional[str] = None,
    tokenizer_mode: str = "auto",
    skip_tokenizer_init: bool = False,
    trust_remote_code: bool = False,
    allowed_local_media_path: str = "",
    tensor_parallel_size: int = 1,
    dtype: str = "auto",
    quantization: Optional[str] = None,
    revision: Optional[str] = None,
    tokenizer_revision: Optional[str] = None,
    seed: int = 0,
    gpu_memory_utilization: float = 0.9,
    swap_space: float = 4,
    cpu_offload_gb: float = 0,
    enforce_eager: Optional[bool] = None,
    max_seq_len_to_capture: int = 8192,
    disable_custom_all_reduce: bool = False,
    disable_async_output_proc: bool = False,
    mm_processor_kwargs: Optional[Dict[str, Any]] = None,
    # After positional args are removed, move this right below `model`
    task = "auto",
    pooling_type: Optional[str] = None,
    pooling_norm: Optional[bool] = None,
    pooling_softmax: Optional[bool] = None,
    pooling_step_tag_id: Optional[int] = None,
    pooling_returned_token_ids: Optional[List[int]] = None,
    **kwargs,
) -> None:
    '''
    LLM constructor.

    Note: if enforce_eager is unset (enforce_eager is None)
    it defaults to False.
    '''

    if "disable_log_stats" not in kwargs:
        kwargs["disable_log_stats"] = True

    engine_args = EngineArgs(
        model=model,
        task=task,
        tokenizer=tokenizer,
        tokenizer_mode=tokenizer_mode,
        skip_tokenizer_init=skip_tokenizer_init,
        trust_remote_code=trust_remote_code,
        allowed_local_media_path=allowed_local_media_path,
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtype,
        quantization=quantization,
        revision=revision,
        tokenizer_revision=tokenizer_revision,
        seed=seed,
        gpu_memory_utilization=gpu_memory_utilization,
        swap_space=swap_space,
        cpu_offload_gb=cpu_offload_gb,
        enforce_eager=enforce_eager,
        max_seq_len_to_capture=max_seq_len_to_capture,
        disable_custom_all_reduce=disable_custom_all_reduce,
        disable_async_output_proc=disable_async_output_proc,
        mm_processor_kwargs=mm_processor_kwargs,
        pooling_type=pooling_type,
        pooling_norm=pooling_norm,
        pooling_softmax=pooling_softmax,
        pooling_step_tag_id=pooling_step_tag_id,
        pooling_returned_token_ids=pooling_returned_token_ids,
        **kwargs,
    )
    return engine_args


def create_model_input_decode() -> ModelInputForHPUWithSamplingMetadata:
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=128
    )

    input_tokens = torch.tensor([[386], [4423], [1101], [1618]], device='hpu:0')
    input_positions = torch.tensor([[6], [8], [6], [6]], device='hpu:0')
    seq_lens = []
    query_lens = []
    lora_mapping = None
    lora_requests = set()

    block_list = torch.zeros(128, dtype=torch.int32, device='hpu:0')
    block_list[:4] = torch.tensor([1, 2, 3, 4,], device='hpu:0')
    block_mapping = torch.zeros(128, dtype=torch.int32, device='hpu:0') - 1
    block_mapping[:4] = torch.tensor([0, 1, 2, 3], device='hpu:0')
    block_usage = torch.zeros(128, dtype=torch.bfloat16, device='hpu:0') + 1
    block_usage[:4] = torch.tensor([7., 9., 7., 7.], device='hpu:0')
    block_indices = torch.tensor([1, 2, 3, 4], device='hpu:0')
    block_offsets = torch.tensor([6, 8, 6, 6], device='hpu:0')
    block_scales = torch.zeros(128, dtype=torch.bfloat16, device='hpu:0') + 1
    block_scales[:4] = torch.tensor([1., 1., 1., 1.], device='hpu:0')

    attn_metadata = HPUAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decode_tokens=30,
        slot_mapping=torch.tensor([[134], [264], [390], [518]], device='hpu:0'),
        multi_modal_placeholder_index_maps=None,
        block_list=block_list,
        block_mapping=block_mapping,
        block_usage=block_usage,
        block_indices=block_indices,
        block_offsets=block_offsets,
        block_scales=block_scales,
        is_prompt=False,
        attn_bias=None,
        seq_lens_tensor=None,
    )

    model_input = ModelInputForHPUWithSamplingMetadata(
        input_tokens=input_tokens,
        input_positions=input_positions,
        seq_lens=seq_lens,
        query_lens=query_lens,
        lora_mapping=lora_mapping,
        lora_requests=lora_requests,
        attn_metadata=attn_metadata,
        multi_modal_kwargs={},
        real_batch_size=4,
        batch_size_padded=4,
        virtual_engine=0,
        lora_ids=[0, 0, 0, 0],
        async_callback=None,
        sampling_metadata=SamplingMetadata(
            seq_groups=[
                SequenceGroupToSample(seq_ids=[0],
                                      sampling_params=sampling_params,
                                      seq_data={0: SequenceData(_prompt_token_ids=array('l', [128000, 9906, 11, 856, 836, 374]),
                                                                                        _output_token_ids=array('l', [386]),
                                                                                        _num_computed_tokens=6)},
                                      seq_len=None,
                                      query_len=1,
                                      generator=None,
                                      is_prompt=False,
                                      prompt_logprob_indices=[],
                                      sample_indices=[0]
                                      ),
                SequenceGroupToSample(seq_ids=[1],
                                      sampling_params=sampling_params,
                                      seq_data={1: SequenceData(_prompt_token_ids=array('l', [128000, 791, 4872, 315, 279, 3723, 4273, 374]),
                                                                                        _output_token_ids=array('l', [4423]),
                                                                                        _num_computed_tokens=8)},
                                      seq_len=None,
                                      query_len=1,
                                      generator=None,
                                      is_prompt=False,
                                      prompt_logprob_indices=[],
                                      sample_indices=[1]
                                      ),
                SequenceGroupToSample(seq_ids=[2],
                                      sampling_params=sampling_params,
                                      seq_data={2: SequenceData(_prompt_token_ids=array('l', [128000, 791, 6864, 315, 9822, 374]),
                                                                                        _output_token_ids=array('l', [1101]),
                                                                                        _num_computed_tokens=6)},
                                      seq_len=None,
                                      query_len=1,
                                      generator=None,
                                      is_prompt=False,
                                      prompt_logprob_indices=[],
                                      sample_indices=[2]
                                      ),
                SequenceGroupToSample(seq_ids=[3],
                                      sampling_params=sampling_params,
                                      seq_data={3: SequenceData(_prompt_token_ids=array('l', [128000, 791, 3938, 315, 15592, 374]),
                                                                                        _output_token_ids=array('l', [1618]),
                                                                                        _num_computed_tokens=6)},
                                      seq_len=None,
                                      query_len=1,
                                      generator=None,
                                      is_prompt=False,
                                      prompt_logprob_indices=[],
                                      sample_indices=[3]
                                      ),
            ],
            selected_token_indices=torch.tensor([0, 1, 2, 3], device='hpu:0'),
            categorized_sample_indices={
                SamplingType.GREEDY: torch.tensor([], device='hpu:0', dtype=torch.int32),
                SamplingType.RANDOM: torch.tensor([0, 1, 2, 3], device='hpu:0', dtype=torch.int32),
                SamplingType.RANDOM_SEED: torch.tensor([], device='hpu:0', dtype=torch.int32)
            },
            num_prompts=0,
            skip_sampler_cpu_output=False,
            reuse_sampling_tensors=False
        )
    )
    return model_input


def create_model_input_prompt() -> ModelInputForHPUWithSamplingMetadata:
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=128
    )
    input_token_s1 = torch.zeros(128, dtype=torch.int64, device='hpu:0')
    input_token_s1[:6] = torch.tensor([128000, 9906, 11, 856, 836, 374], device='hpu:0')
    input_token_s2 = torch.zeros(128, dtype=torch.int64, device='hpu:0')
    input_token_s2[:8] = torch.tensor([128000, 791, 4872, 315, 279, 3723, 4273, 374], device='hpu:0')
    input_token_s3 = torch.zeros(128, dtype=torch.int64, device='hpu:0')
    input_token_s3[:6] = torch.tensor([128000, 791, 6864, 315, 9822, 374], device='hpu:0')
    input_token_s4 = torch.zeros(128, dtype=torch.int64, device='hpu:0')
    input_token_s4[:6] = torch.tensor([128000, 791, 3938, 315, 15592, 374], device='hpu:0')

    input_positions_1 = torch.zeros(128, dtype=torch.int64, device='hpu:0')
    input_positions_1[:6] = torch.tensor([0, 1, 2, 3, 4, 5], device='hpu:0')
    input_positions_2 = torch.zeros(128, dtype=torch.int64, device='hpu:0')
    input_positions_2[:8] = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], device='hpu:0')
    input_positions_3 = torch.zeros(128, dtype=torch.int64, device='hpu:0')
    input_positions_3[:6] = torch.tensor([0, 1, 2, 3, 4, 5], device='hpu:0')
    input_positions_4 = torch.zeros(128, dtype=torch.int64, device='hpu:0')
    input_positions_4[:6] = torch.tensor([0, 1, 2, 3, 4, 5], device='hpu:0')

    slot_mapping_1 = torch.zeros(128, dtype=torch.int64, device='hpu:0')
    slot_mapping_1[:6] = torch.tensor([128, 129, 130, 131, 132, 133], device='hpu:0')
    slot_mapping_2 = torch.zeros(128, dtype=torch.int64, device='hpu:0')
    slot_mapping_2[:8] = torch.tensor([256, 257, 258, 259, 260, 261, 262, 263], device='hpu:0')
    slot_mapping_3 = torch.zeros(128, dtype=torch.int64, device='hpu:0')
    slot_mapping_3[:6] = torch.tensor([384, 385, 386, 387, 388, 389], device='hpu:0')
    slot_mapping_4 = torch.zeros(128, dtype=torch.int64, device='hpu:0')
    slot_mapping_4[:6] = torch.tensor([512, 513, 514, 515, 516, 517], device='hpu:0')

    attn_metadata = HPUAttentionMetadata(
            num_prefills=4, 
            num_prefill_tokens=26, 
            num_decode_tokens=0, 
            slot_mapping=torch.stack([slot_mapping_1, slot_mapping_2, slot_mapping_3, slot_mapping_4]),
            multi_modal_placeholder_index_maps=None,
            block_list=None,
            block_mapping=None,
           block_usage=None,
           block_indices=torch.tensor([1, 2, 3, 4], device='hpu:0'),
           block_offsets=None,
           block_scales=None,
           is_prompt=True, 
           attn_bias=None, 
           seq_lens_tensor=torch.tensor([6, 8, 6, 6], device='hpu:0')
           )

    model_input = ModelInputForHPUWithSamplingMetadata(
        input_tokens=torch.stack([input_token_s1, input_token_s2, input_token_s3, input_token_s4]),    
        input_positions=torch.stack([input_positions_1, input_positions_2, input_positions_3, input_positions_4]),
        seq_lens=[6, 8, 6, 6],
        query_lens=[6, 8, 6, 6],
        lora_mapping=None,
        lora_requests=set(),
        attn_metadata=attn_metadata,
        multi_modal_kwargs={}, 
        real_batch_size=4, 
        batch_size_padded=4, 
        virtual_engine=0, 
        lora_ids=[0, 0, 0, 0], 
        async_callback=None,
        sampling_metadata=SamplingMetadata(
            seq_groups=[SequenceGroupToSample(seq_ids=[0], 
                                                seq_data={0: SequenceData(_prompt_token_ids=array('l', [128000, 9906, 11, 856, 836, 374]))}, 
                                            sampling_params=sampling_params,
                                            seq_len=6,
                                            query_len=6,
                                            generator=None,
                                            is_prompt=True,
                                            prompt_logprob_indices=[],
                                            sample_indices=[0]
                                            ),
                        SequenceGroupToSample(seq_ids=[1], 
                                                seq_data={1: SequenceData(_prompt_token_ids=array('l', [128000, 791, 4872, 315, 279, 3723, 4273, 374]))}, 
                                                sampling_params=sampling_params,
                                                seq_len=8,
                                                query_len=8,
                                                generator=None,
                                                is_prompt=True,
                                                prompt_logprob_indices=[],
                                                sample_indices=[1]
                                                ),
                        SequenceGroupToSample(seq_ids=[2], 
                                                seq_data={2: SequenceData(_prompt_token_ids=array('l', [128000, 791, 6864, 315, 9822, 374]))}, 
                                                sampling_params=sampling_params,
                                                seq_len=6,
                                                query_len=6,
                                                generator=None,
                                                is_prompt=True,
                                                prompt_logprob_indices=[],
                                                sample_indices=[2]
                                                ),
                        SequenceGroupToSample(seq_ids=[3], 
                                                seq_data={3: SequenceData(_prompt_token_ids=array('l', [128000, 791, 3938, 315, 15592, 374]))}, 
                                                sampling_params=sampling_params,
                                                seq_len=6,
                                                query_len=6,
                                                generator=None,
                                                is_prompt=True,
                                                prompt_logprob_indices=[],
                                                sample_indices=[3]
                                                )],
            selected_token_indices=torch.tensor([5, 135, 261, 389], device='hpu:0'),
            categorized_sample_indices={
                SamplingType.GREEDY: torch.tensor([], device='hpu:0', dtype=torch.int32),
                SamplingType.RANDOM: torch.tensor([0, 1, 2, 3], device='hpu:0', dtype=torch.int32),
                SamplingType.RANDOM_SEED: torch.tensor([], device='hpu:0', dtype=torch.int32)
            },
            num_prompts=4,
            skip_sampler_cpu_output=False,
            reuse_sampling_tensors=False
        )
    )
    return model_input

def main():
    
    engine_args = create_engine_args(
        model="meta-llama/Llama-3.2-1B", device="hpu", dtype="bfloat16"
    )
    vllm_config = engine_args.create_engine_config()

    distributed_init_method = get_distributed_init_method(
        get_ip(), get_open_port())

    # Initialize HPU worker
    worker = HPUWorker(
        vllm_config=vllm_config,
        local_rank=0,
        rank=0,
        distributed_init_method=distributed_init_method,
        is_driver_worker=True
    )

    # # Initialize device and load model
    worker.init_device()
    worker.load_model()

    # Sample prompts
    prompt = "Hello, my name is"

    # Create sampling parameters

    num_gpu_blocks, num_cpu_blocks = worker.determine_num_available_blocks()
    worker.initialize_cache(num_gpu_blocks, num_cpu_blocks)

    worker_input = WorkerInput(
        num_seq_groups=4,
        blocks_to_swap_in=torch.tensor([], device='hpu:0', dtype=torch.int64).view(-1, 2),
        blocks_to_swap_out=torch.tensor([], device='hpu:0', dtype=torch.int64).view(-1, 2),
        blocks_to_copy=torch.tensor([], device='hpu:0', dtype=torch.int64).view(-1, 2),
        virtual_engine=0,
        num_steps=1
    )

    # worker.execute_worker(worker_input)

    model_input = create_model_input_prompt()

    output = worker.model_runner.execute_model(
        model_input=model_input,
        kv_caches=worker.kv_cache[worker_input.virtual_engine] if worker.kv_cache is not None else None,
        intermediate_tensors=None,
        num_steps=worker_input.num_steps,
        **{}
    )

    print(output)

    model_input = create_model_input_decode()

    output = worker.model_runner.execute_model(
        model_input=model_input,
        kv_caches=worker.kv_cache[worker_input.virtual_engine] if worker.kv_cache is not None else None,
        intermediate_tensors=None,
        num_steps=worker_input.num_steps,
        **{}
    )

    print(output)

    # outputs = worker.execute_model(
    #     execute_model_req=worker._prepare_worker_input(seq_group_metadata_list)
    # )

    # print(outputs)

if __name__ == "__main__":
    main() 