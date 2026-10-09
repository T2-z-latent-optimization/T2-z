"""Saved-tensor storage for a differentiable TTT stream.

Keep existing model parameter storage on CUDA, cap additional retained CUDA
storage, and offload other saved tensors to pinned host memory. Repeated saves
of the same unmodified tensor view share a snapshot. No fast state is detached
from the differentiable update graph; only autograd's saved values are packed.
"""
from __future__ import annotations

from dataclasses import dataclass
import weakref

import torch


@dataclass
class HostSnapshot:
    tensor: torch.Tensor
    device: torch.device
    ready: torch.cuda.Event


class SelectiveSavedTensorStorage:
    def __init__(self, model, *, gpu_budget_gib=12.0, host_budget_gib=None, pin_memory=True):
        self.gpu_budget = int(float(gpu_budget_gib) * 1024 ** 3)
        if self.gpu_budget < 0:
            raise ValueError("Saved-tensor CUDA budget must be nonnegative.")
        self.host_budget = None if host_budget_gib is None else int(float(host_budget_gib) * 1024 ** 3)
        if self.host_budget is not None and self.host_budget < 0:
            raise ValueError("Saved-tensor host budget must be nonnegative.")
        self.pin_memory = bool(pin_memory)
        self.parameters = {
            (tensor.device, tensor.untyped_storage().data_ptr())
            for tensor in (*model.parameters(), *model.buffers())
            if tensor.device.type == "cuda" and tensor.numel()
        }
        self.cache = {}
        self.gpu_storages = set()
        self.gpu_bytes = 0
        self.statistics = {}
        self.begin_update()

    def begin_update(self):
        self.end_stream()
        self.statistics = {
            "parameter_saves_kept_on_gpu": 0,
            "snapshot_cache_hits": 0,
            "host_snapshot_count": 0,
            "host_snapshot_bytes": 0,
            "host_snapshot_peak_bytes": 0,
            "host_restore_count": 0,
            "host_restore_bytes": 0,
            "extra_gpu_storage_peak_bytes": 0,
        }

    def begin_stream(self):
        self.end_stream()

    def end_stream(self):
        self.cache.clear()
        self.gpu_storages.clear()
        self.gpu_bytes = 0
        self.host_bytes = 0

    def scope(self):
        return torch.autograd.graph.saved_tensors_hooks(self.pack, self.unpack)

    def pack(self, tensor):
        if tensor.device.type != "cuda" or not tensor.numel():
            return tensor.detach()
        storage = tensor.untyped_storage()
        storage_key = (tensor.device, storage.data_ptr())
        if storage_key in self.parameters:

            self.statistics["parameter_saves_kept_on_gpu"] += 1
            return tensor.detach()
        key = (
            storage_key, tensor.storage_offset(), tuple(tensor.shape),
            tuple(tensor.stride()), tensor.dtype, tensor._version,
            tensor.is_conj(), tensor.is_neg(),
        )
        cached = self.cache.get(key)
        if cached is not None and cached[0]() is not None:
            self.statistics["snapshot_cache_hits"] += 1
            return cached[1]
        storage_bytes = storage.nbytes()
        if storage_key in self.gpu_storages or self.gpu_bytes + storage_bytes <= self.gpu_budget:
            if storage_key not in self.gpu_storages:
                self.gpu_storages.add(storage_key)
                self.gpu_bytes += storage_bytes
                self.statistics["extra_gpu_storage_peak_bytes"] = max(
                    self.statistics["extra_gpu_storage_peak_bytes"], self.gpu_bytes,
                )
            packed = tensor.detach()
        else:
            host_bytes = tensor.numel() * tensor.element_size()
            if self.host_budget is not None and self.host_bytes + host_bytes > self.host_budget:
                raise MemoryError(
                    "TTT saved-tensor host budget exceeded before allocating another snapshot: "
                    f"live={self.host_bytes / 1024**3:.2f} GiB, "
                    f"request={host_bytes / 1024**3:.2f} GiB, "
                    f"limit={self.host_budget / 1024**3:.2f} GiB per rank. "
                    "Use segmented pure fast-state recomputation rather than increasing host RAM."
                )
            stream = torch.cuda.current_stream(tensor.device)
            host = torch.empty_like(tensor, device="cpu", pin_memory=self.pin_memory, requires_grad=False)
            host.copy_(tensor.detach(), non_blocking=True)
            tensor.record_stream(stream)
            ready = torch.cuda.Event()
            ready.record(stream)
            packed = HostSnapshot(host, tensor.device, ready)
            self.statistics["host_snapshot_count"] += 1
            self.statistics["host_snapshot_bytes"] += tensor.numel() * tensor.element_size()
            self.host_bytes += host_bytes
            self.statistics["host_snapshot_peak_bytes"] = max(
                self.statistics["host_snapshot_peak_bytes"], self.host_bytes,
            )


        self.cache[key] = (weakref.ref(tensor), packed)
        return packed

    def unpack(self, packed):
        if isinstance(packed, torch.Tensor):
            return packed
        stream = torch.cuda.current_stream(packed.device)
        stream.wait_event(packed.ready)
        self.statistics["host_restore_count"] += 1
        self.statistics["host_restore_bytes"] += packed.tensor.numel() * packed.tensor.element_size()
        return packed.tensor.to(packed.device, non_blocking=True)
