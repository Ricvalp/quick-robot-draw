"""
Utilities for collating QuickDraw episodes into SketchRNN-friendly batches.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

__all__ = ["LSTMCollator"]


class LSTMCollator:
    """
    Convert sampled QuickDraw episodes into padded stroke tensors used by SketchRNN.

    Every returned sample is a tensor shaped ``(T, 5)`` with ``(Δx, Δy, p1, p2, p3)``
    channels plus an explicit EOS token. The collator extracts the query sketch from
    the episode, filters out special control tokens, converts coordinates into deltas
    when required, and appends the `[0, 0, 0, 0, 1]` sentinel.
    """

    def __init__(
        self,
        *,
        max_seq_len: Optional[int] = 300,
        coordinate_mode: str = "delta",
        pad_value: float = 0.0,
    ) -> None:
        if max_seq_len is not None and max_seq_len < 2:
            raise ValueError("max_seq_len must be >= 2 when provided.")
        coord = coordinate_mode.lower()
        if coord not in {"delta", "absolute"}:
            raise ValueError("coordinate_mode must be 'delta' or 'absolute'.")
        self.max_seq_len = max_seq_len
        self.coordinate_mode = coord
        self.pad_value = pad_value

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        sequences: List[torch.Tensor] = []
        lengths: List[int] = []

        for sample in batch:
            tokens = sample["tokens"]
            metadata = sample.get("metadata", {})
            query = self._extract_query(tokens, metadata)
            if query is None or query.shape[0] == 0:
                continue
            strokes = self._tokens_to_strokes(query)
            if strokes.shape[0] < 2:
                # Need at least one actual point plus EOS.
                continue
            if self.max_seq_len is not None and strokes.shape[0] > self.max_seq_len:
                strokes = self._truncate(strokes, self.max_seq_len)
            sequences.append(strokes)
            lengths.append(strokes.shape[0])

        if not sequences:
            raise ValueError("No valid query sketches found in batch for LSTMCollator.")

        max_len = max(lengths)
        batch_size = len(sequences)
        padded = torch.full(
            (batch_size, max_len, 5),
            self.pad_value,
            dtype=sequences[0].dtype,
        )
        mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

        for idx, (seq, length) in enumerate(zip(sequences, lengths)):
            padded[idx, :length] = seq
            mask[idx, :length] = True

        return {
            "strokes": padded,
            "lengths": torch.tensor(lengths, dtype=torch.long),
            "mask": mask,
        }

    def _extract_query(
        self,
        tokens: torch.Tensor,
        metadata: Optional[Dict[str, object]],
    ) -> Optional[torch.Tensor]:
        if metadata:
            start = metadata.get("query_start_index")
            length = metadata.get("query_length")
            if start is not None and length is not None and length > 0:
                start_idx = int(start)
                end_idx = start_idx + int(length)
                if end_idx <= tokens.shape[0]:
                    return tokens[start_idx:end_idx].clone()

        reset_idx = self._first_index(tokens[:, 5])
        if reset_idx is None:
            return None
        start_idx = self._first_index(tokens[:, 3], min_index=reset_idx)
        if start_idx is None:
            return None
        stop_idx = self._first_index(tokens[:, 6], min_index=start_idx)
        if stop_idx is None or stop_idx <= start_idx + 1:
            return None
        return tokens[start_idx + 1 : stop_idx].clone()

    @staticmethod
    def _first_index(column: torch.Tensor, min_index: int = -1) -> Optional[int]:
        indices = torch.nonzero(column > 0.5, as_tuple=False)
        for idx in indices:
            value = int(idx.item())
            if value > min_index:
                return value
        return None

    def _tokens_to_strokes(self, tokens: torch.Tensor) -> torch.Tensor:
        coords = tokens[:, :2].to(dtype=torch.float32)
        if self.coordinate_mode == "absolute":
            deltas = torch.zeros_like(coords)
            deltas[0] = coords[0]
            if coords.shape[0] > 1:
                deltas[1:] = coords[1:] - coords[:-1]
        else:
            deltas = coords.clone()

        pen = tokens[:, 2].to(dtype=torch.float32)
        stroke_end = torch.zeros_like(pen, dtype=torch.bool)
        if pen.shape[0] > 1:
            stroke_end[:-1] = pen[1:] < 0.5
        stroke_end[-1] = True

        pen_down = (~stroke_end).float()
        pen_up = stroke_end.float()
        eos = torch.zeros_like(pen_down)

        strokes = torch.stack([deltas[:, 0], deltas[:, 1], pen_down, pen_up, eos], dim=-1)
        strokes = torch.cat([strokes, self._eos_row(strokes.device, strokes.dtype)], dim=0)
        return strokes

    def _truncate(self, strokes: torch.Tensor, max_len: int) -> torch.Tensor:
        truncated = strokes[:max_len].clone()
        truncated[-1] = 0.0
        truncated[-1, -1] = 1.0
        return truncated

    @staticmethod
    def _eos_row(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        row = torch.zeros(1, 5, device=device, dtype=dtype)
        row[0, -1] = 1.0
        return row
