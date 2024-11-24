"""vLLM distributed KV cache transfer API.
These APIs are used in `vllm/worker/model_runner.py`.

Currently supporting TP. The TP between prefill and decode instance needs to be 
the same.

Workflow (disaggregated prefill)
- In prefill instance
    - After prefill, vLLM `insert` its KV caches into a lookup buffer.
    - The prefill instance will also open up a thread that listens to 
      `drop_select` request.
- In decode instance
    - vLLM first runs `drop_select` to send input tokens and a mask on input 
      tokens (we call it roi, region of interest) to prefill instance
    - The prefill instance then respond to `drop_select` request by
        - Finding a match in current lookup buffer.
        - Clone and send the matched item out
        - Delete the matched item in the lookup buffer to free up GPU memory.
    - The decode vLLM then store the KV cache into paged memory.
"""
from typing import TYPE_CHECKING, List, Tuple, Union

if TYPE_CHECKING:
    from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata

import torch

from vllm.distributed.kv_transfer.kv_connector.factory import (
    KVConnectorFactory)
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors

logger = init_logger(__name__)


class KVTransferAgent:
    """
    A class designated for distributed KV transfer
    
    Target use cases:
        1. Disaggregated prefill
        2. Remote KV cache storage
    """

    def __init__(
        self,
        rank: int,
        local_rank: int,
        config,
    ):

        self.config = config
        assert self.config.is_kv_transfer_instance, "KV cache transfer "\
            "agent should only be used when kv_connector is set."

        self.connector = KVConnectorFactory.create_connector(
            rank, local_rank, config)

    def send_kv_caches_and_hidden_states(
        self,
        model_executable: torch.nn.Module,
        input_tokens,
        attn_metadata,
        kv_caches: List[torch.Tensor],
        hidden_or_intermediate_states: Union[torch.Tensor,
                                             IntermediateTensors],
    ) -> None:

        self.connector.send_kv_caches_and_hidden_states(
            model_executable, input_tokens, attn_metadata, kv_caches,
            hidden_or_intermediate_states)

    def close(self) -> None:
        self.connector.close()

    def recv_kv_caches_and_hidden_states(
        self, model_executable: torch.nn.Module,
        input_tokens,
        attn_metadata,
        kv_caches: List[torch.Tensor]
    ) -> Tuple[Union[torch.Tensor, IntermediateTensors], bool,
               "ModelInputForGPUWithSamplingMetadata"]:

        return self.connector.recv_kv_caches_and_hidden_states(
            model_executable, input_tokens, attn_metadata, kv_caches)
        
    def send_single_layer_kv_cache(
        self,
        model_executable: torch.nn.Module,
        input_tokens,
        attn_metadata,
        kv_caches: List[torch.Tensor],
        layer_id
    ) -> None:

        self.connector.send_single_layer_kv_cache(
            model_executable, input_tokens, attn_metadata, kv_caches, layer_id)

    def recv_single_layer_kv_cache(
        self,
        model_executable: torch.nn.Module,
        input_tokens,
        attn_metadata,
        kv_caches: List[torch.Tensor],
        layer_id
    ) -> None:

        self.connector.recv_single_layer_kv_cache(
            model_executable, input_tokens, attn_metadata, kv_caches, layer_id)
    
    def send_hidden_states(
        self,
        input_tokens,
        attn_metadata,
        hidden_or_intermediate_states,
    ):
        self.connector.send_hidden_states(
            input_tokens, attn_metadata, hidden_or_intermediate_states
        )

    def recv_hidden_states(
        self,
        input_tokens,
        attn_metadata,
    ): 
        return self.connector.recv_hidden_states(
            input_tokens, attn_metadata
        )