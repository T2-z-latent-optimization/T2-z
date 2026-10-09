import json
import os
from typing import Iterable, Optional

from diffsynth.core import UnifiedDataset


class RoboTwinUnifiedDataset(UnifiedDataset):
    def __init__(
        self,
        base_path: str,
        metadata_path: str,
        repeat: int = 1,
        data_file_keys: Iterable[str] = ("video", "action"),
        main_data_operator=lambda x: x,
        special_operator_map: Optional[dict] = None,
        sample_indices: Optional[Iterable[int]] = None,
    ):
        self.base_path = base_path
        self.metadata_path = metadata_path
        self.repeat = int(repeat)
        self.data_file_keys = tuple(data_file_keys)
        self.main_data_operator = main_data_operator
        self.special_operator_map = {} if special_operator_map is None else dict(special_operator_map)
        self.sample_indices = None if sample_indices is None else [int(index) for index in sample_indices]
        self.load_from_cache = False
        self.data = self._load_metadata(metadata_path)
        self._apply_sample_selection()
        print(f"Dataset size: {len(self.data)}, repeat: {self.repeat}, total: {len(self)}")

    def _load_metadata(self, metadata_path: str):
        if metadata_path.endswith(".json"):
            with open(metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data

        if metadata_path.endswith(".jsonl"):
            rows = []
            with open(metadata_path, "r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    text = line.strip()
                    if not text:
                        continue
                    row = json.loads(text)
                    rows.append(row)
            return rows

    def _apply_sample_selection(self):
        if self.sample_indices is None:
            return
        invalid = [index for index in self.sample_indices if index < 0 or index >= len(self.data)]
        if invalid:
            raise IndexError(f"Sample indices out of range: {invalid} (dataset size: {len(self.data)})")
        self.data = [self.data[index] for index in self.sample_indices]

    def __len__(self):
        return len(self.data) * self.repeat

    def __getitem__(self, data_id: int):
        data = self.data[int(data_id) % len(self.data)].copy()
        for key in self.data_file_keys:
            if key in self.special_operator_map:
                source = data[key] if key in data else data
                data[key] = self.special_operator_map[key](self._wrap_frame_range_metadata(data, source))
            elif key in data:
                data[key] = self.main_data_operator(self._wrap_frame_range_metadata(data, data[key]))
        return data

    def _resolve_frame_range(self, data: dict):
        start_frame = int(data.get("start_frame", 0))
        if data.get("end_frame") is not None:
            end_frame = int(data["end_frame"])
        elif data.get("length") is not None:
            end_frame = start_frame + int(data["length"]) - 1
        return start_frame, end_frame

    def _wrap_frame_range_metadata(self, data, payload):
        if not isinstance(data, dict):
            return payload

        start_frame, end_frame = self._resolve_frame_range(data)

        def wrap_item(item):
            if isinstance(item, str):
                return {"data": item, "start_frame": start_frame, "end_frame": end_frame}
            if isinstance(item, dict):
                wrapped = item.copy()
                wrapped.setdefault("start_frame", start_frame)
                wrapped.setdefault("end_frame", end_frame)
                return wrapped
            return item

        if isinstance(payload, (list, tuple)):
            return [wrap_item(item) for item in payload]
        return wrap_item(payload)
