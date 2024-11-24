def need_recv_kv(vllm_config, attn_metadata, kv_caches) -> bool:
    """Check if we need to receive kv-cache from the other worker.
    We need to receive KV when
        1. current vLLM instance is KV cache consumer/decode vLLM instance
        2. this batch is not a profiling run
        3. this batch is a prefill run
        
    Args:
        model_input: input to the model executable
        kv_caches: vLLM's paged memory
    """

    prefill_meta = attn_metadata.prefill_metadata

    # check if the current run is profiling
    is_profile_run = (kv_caches[0].numel() == 0)
    # check if the current run is prefill
    is_prefill_run = prefill_meta is not None

    if vllm_config.kv_transfer_config is None:
        return False

    return vllm_config.kv_transfer_config.is_kv_consumer and (
        not is_profile_run) and is_prefill_run

def need_send_kv(vllm_config, attn_metadata, kv_caches) -> bool:
    """Check if we need to send kv-cache to the other worker.
    We need to send KV when
        1. current vLLM instance is KV cache producer/prefill vLLM instance
        2. this batch is not a profiling run
        3. this batch is a prefill run
        
    Args:
        model_input: input to the model executable
        kv_caches: vLLM's paged memory
    """

    prefill_meta = attn_metadata.prefill_metadata

    # check if the current run is profiling
    is_profile_run = (kv_caches[0].numel() == 0)
    # check if the current run is prefill
    is_prefill_run = prefill_meta is not None

    if vllm_config.kv_transfer_config is None:
        return False

    return vllm_config.kv_transfer_config.is_kv_producer and (
        not is_profile_run) and is_prefill_run