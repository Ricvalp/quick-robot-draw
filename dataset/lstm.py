"""
Utilities for collating QuickDraw episodes into SketchRNN-friendly batches.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

__all__ = ["SketchRNNCollator", "InContextSketchRNNCollator"]


# class LSTMCollator:
#     """
#     Convert sampled QuickDraw episodes into padded stroke tensors used by SketchRNN.

#     Every returned sample is a tensor shaped ``(T, 5)`` with ``(Δx, Δy, p1, p2, p3)``
#     channels plus an explicit EOS token. The collator extracts the query sketch from
#     the episode, filters out special control tokens, converts coordinates into deltas
#     when required, and appends the `[0, 0, 0, 0, 1]` sentinel.
#     """

#     def __init__(
#         self,
#         *,
#         max_seq_len: Optional[int] = 300,
#         pad_value: float = 0.0,
#     ) -> None:
#         if max_seq_len is not None and max_seq_len < 2:
#             raise ValueError("max_seq_len must be >= 2 when provided.")
#         self.max_seq_len = max_seq_len
#         self.pad_value = pad_value

#     def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
#         sequences: List[torch.Tensor] = []
#         lengths: List[int] = []

#         for sample in batch:
#             tokens = sample["tokens"]
#             metadata = sample.get("metadata", {})
#             query = self._extract_query(tokens, metadata)
#             if query is None or query.shape[0] == 0:
#                 continue
#             strokes = self._tokens_to_strokes(query)
#             if strokes.shape[0] < 2:
#                 # Need at least one actual point plus EOS.
#                 continue
#             if self.max_seq_len is not None and strokes.shape[0] > self.max_seq_len:
#                 strokes = self._truncate(strokes, self.max_seq_len)
#             sequences.append(strokes)
#             lengths.append(strokes.shape[0])

#         if not sequences:
#             raise ValueError("No valid query sketches found in batch for LSTMCollator.")

#         max_len = max(lengths)
#         batch_size = len(sequences)
#         padded = torch.full(
#             (batch_size, max_len, 5),
#             self.pad_value,
#             dtype=sequences[0].dtype,
#         )
#         mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

#         for idx, (seq, length) in enumerate(zip(sequences, lengths)):
#             padded[idx, :length] = seq
#             mask[idx, :length] = True

#         return {
#             "strokes": padded,
#             "lengths": torch.tensor(lengths, dtype=torch.long),
#             "mask": mask,
#         }

#     def _extract_query(
#         self,
#         tokens: torch.Tensor,
#         metadata: Optional[Dict[str, object]],
#     ) -> Optional[torch.Tensor]:
#         if metadata:
#             start = metadata.get("query_start_index")
#             length = metadata.get("query_length")
#             if start is not None and length is not None and length > 0:
#                 start_idx = int(start)
#                 end_idx = start_idx + int(length)
#                 if end_idx <= tokens.shape[0]:
#                     return tokens[start_idx:end_idx].clone()

#         print("WARNING: Falling back to automatic query extraction. Never tested!!!")
#         reset_idx = self._first_index(tokens[:, 5])
#         if reset_idx is None:
#             return None
#         start_idx = self._first_index(tokens[:, 3], min_index=reset_idx)
#         if start_idx is None:
#             return None
#         stop_idx = self._first_index(tokens[:, 6], min_index=start_idx)
#         if stop_idx is None or stop_idx <= start_idx + 1:
#             return None
#         return tokens[start_idx + 1 : stop_idx].clone()

#     @staticmethod
#     def _first_index(column: torch.Tensor, min_index: int = -1) -> Optional[int]:
#         indices = torch.nonzero(column > 0.5, as_tuple=False)
#         for idx in indices:
#             value = int(idx.item())
#             if value > min_index:
#                 return value
#         return None

#     def _tokens_to_strokes(self, tokens: torch.Tensor) -> torch.Tensor:
#         coords = tokens[:, :2].to(dtype=torch.float32)

#         deltas = coords.clone()

#         pen = tokens[:, 2].to(dtype=torch.float32)
#         stroke_end = torch.zeros_like(pen, dtype=torch.bool)
#         if pen.shape[0] > 1:
#             stroke_end[:-1] = pen[1:] < 0.5
#         stroke_end[-1] = True

#         pen_down = (~stroke_end).float()
#         pen_up = stroke_end.float()
#         eos = torch.zeros_like(pen_down)

#         strokes = torch.stack(
#             [deltas[:, 0], deltas[:, 1], pen_down, pen_up, eos], dim=-1
#         )
#         strokes = torch.cat(
#             [strokes, self._eos_row(strokes.device, strokes.dtype)], dim=0
#         )
#         return strokes

#     def _truncate(self, strokes: torch.Tensor, max_len: int) -> torch.Tensor:
#         truncated = strokes[:max_len].clone()
#         truncated[-1] = 0.0
#         truncated[-1, -1] = 1.0
#         return truncated

#     @staticmethod
#     def _eos_row(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
#         row = torch.zeros(1, 5, device=device, dtype=dtype)
#         row[0, -1] = 1.0
#         return row


class SketchRNNCollator:
    """
    Collator that produces SketchRNN-style sequences.

    Output:
        strokes: (B, Nmax, 5) with (Δx, Δy, p1, p2, p3)
                 padded with pure EOS tokens (0, 0, 0, 0, 1).
        lengths: (B,) true lengths *including* the first EOS,
                 but excluding padding.
        mask:    (B, Nmax) True for real (non-padding) time steps.

    Assumptions:
        - Input `tokens` contain at least columns:
            0: x
            1: y
            2: pen-down-ish flag in [0, 1]
        - Stroke boundaries can be inferred from column 2:
            a stroke ends when next pen < 0.5, or at last point.
        - `max_seq_len` is set to dataset-wide Nmax (#steps including EOS).
    """

    def __init__(
        self,
        *,
        max_seq_len: Optional[int],  # <-- set this to dataset-wide Nmax for exact match
        tokens_are_deltas: bool = True,
    ) -> None:
        if max_seq_len is not None and max_seq_len < 2:
            raise ValueError("max_seq_len must be >= 2 when provided.")
        self.max_seq_len = max_seq_len
        self.tokens_are_deltas = tokens_are_deltas

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        sequences: List[torch.Tensor] = []
        lengths: List[int] = []

        for sample in batch:
            tokens = sample["tokens"]
            metadata = sample.get("metadata", {})
            query = self._extract_query(tokens, metadata)
            if query is None or query.shape[0] == 0:
                continue

            strokes = self._tokens_to_sketchrnn_strokes(query)
            # Need at least one actual point + EOS
            if strokes.shape[0] < 2:
                continue

            # Optional truncation to max_seq_len (if sequences can exceed Nmax)
            if self.max_seq_len is not None and strokes.shape[0] > self.max_seq_len:
                strokes = self._truncate_with_eos(strokes, self.max_seq_len)

            sequences.append(strokes)
            lengths.append(strokes.shape[0])

        if not sequences:
            raise ValueError(
                "No valid query sketches found in batch for SketchRNNCollator."
            )

        if self.max_seq_len is not None:
            max_len = self.max_seq_len  # dataset-wide Nmax
        else:
            # This deviates from the paper; ok for debugging, but not exact.
            max_len = max(lengths)

        batch_size = len(sequences)
        dtype = sequences[0].dtype
        device = sequences[0].device

        # Fill everything with EOS = (0, 0, 0, 0, 1)
        eos_row = torch.tensor([0, 0, 0, 0, 1], dtype=dtype, device=device)
        padded = eos_row.view(1, 1, 5).expand(batch_size, max_len, 5).clone()

        # mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device)

        for idx, (seq, length) in enumerate(zip(sequences, lengths)):
            L = min(length, max_len)
            padded[idx, :L] = seq[:L]
            # mask[idx, :L] = True

        return {
            "strokes": padded,
            "lengths": torch.tensor(lengths, dtype=torch.long, device=device),
            # "mask": mask,
        }

    # --- Query extraction: kept exactly as you already had it ---
    def _extract_query(
        self,
        tokens: torch.Tensor,
        metadata: Optional[Dict[str, object]],
    ) -> Optional[torch.Tensor]:
        if metadata:
            start = metadata.get("query_start_index")
            length = metadata.get("query_length")
            if start is not None and length is not None and length > 0:
                start_idx = int(start) - 1
                end_idx = start_idx + int(length) + 1
                if end_idx <= tokens.shape[0]:
                    return tokens[start_idx:end_idx].clone()

        print("WARNING: Falling back to automatic query extraction. Never tested!!!")
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

    # --- This is the core: convert your tokens into SketchRNN strokes ---
    def _tokens_to_sketchrnn_strokes(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Convert a (T, D) token tensor into a (T+1, 5) SketchRNN stroke sequence.

        Steps:
            1. Take coords (x, y).
            2. Convert to deltas (Δx, Δy) if needed.
            3. Infer stroke boundaries from pen column.
            4. Build (Δx, Δy, p1, p2, p3=0) for all real points.
            5. Append a single EOS row (0, 0, 0, 0, 1).
        """
        # 1) coords
        coords = tokens[:, :2].to(dtype=torch.float32)

        # 2) deltas (SketchRNN uses deltas, not absolute coords)
        if self.tokens_are_deltas:
            deltas = coords
        else:
            deltas = torch.zeros_like(coords)
            if coords.shape[0] > 1:
                deltas[1:] = coords[1:] - coords[:-1]
            # deltas[0] = (0, 0): first move is relative to origin

        # 3) infer stroke endings from pen column
        pen = tokens[:, 2].to(dtype=torch.float32)  # 1 while drawing, 0 otherwise

        pen_down = pen
        pen_up = 1.0 - pen_down

        # p3 = EOS; only used for explicit EOS + padding
        eos = torch.zeros_like(pen_down)

        strokes = torch.stack(
            [deltas[:, 0], deltas[:, 1], pen_down, pen_up, eos],
            dim=-1,
        )

        # 5) append explicit EOS row (0, 0, 0, 0, 1)
        eos_row = self._eos_row(device=strokes.device, dtype=strokes.dtype)
        strokes = torch.cat([strokes, eos_row], dim=0)

        return strokes

    def _truncate_with_eos(self, strokes: torch.Tensor, max_len: int) -> torch.Tensor:
        """
        Truncate to max_len but ensure last row is EOS (0, 0, 0, 0, 1).

        This is important to keep exactly one explicit EOS in truncated sequences.
        """
        truncated = strokes[:max_len].clone()
        # overwrite last row with a clean EOS row
        truncated[-1] = 0.0
        truncated[-1, -1] = 1.0
        return truncated

    @staticmethod
    def _eos_row(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        row = torch.zeros(1, 5, device=device, dtype=dtype)
        row[0, -1] = 1.0
        return row


class InContextSketchRNNCollator:
    """
    TODO: Docstring for InContextSketchRNNCollator.
    """

    def __init__(
        self,
        *,
        max_query_len: Optional[int],
        max_context_len: Optional[
            int
        ],  # <-- set this to dataset-wide Nmax for exact match
        tokens_are_deltas: bool = True,
    ) -> None:
        if max_query_len is not None and max_query_len < 2:
            raise ValueError("max_query_len must be >= 2 when provided.")
        if max_context_len is not None and max_context_len < 2:
            raise ValueError("max_context_len must be >= 2 when provided.")
        self.max_query_len = max_query_len
        self.max_context_len = max_context_len
        self.tokens_are_deltas = tokens_are_deltas

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        queries: List[torch.Tensor] = []
        queries_lengths: List[int] = []
        contexts: List[torch.Tensor] = []
        contexts_lengths: List[int] = []

        for sample in batch:
            tokens = sample["tokens"]
            # metadata = sample.get("metadata", {})
            context, query = self._extract_context_query(tokens)

            # strokes = self._tokens_to_sketchrnn_strokes(query)

            # Optional truncation to max_seq_len (if sequences can exceed Nmax)
            # if self.max_seq_len is not None and query.shape[0] > self.max_seq_len:
            #     strokes = self._truncate_with_eos(strokes, self.max_seq_len)

            queries.append(query)
            queries_lengths.append(query.shape[0])

            contexts.append(context)
            contexts_lengths.append(context.shape[0])

        if not queries:
            raise ValueError(
                "No valid query sketches found in batch for SketchRNNCollator."
            )

        if self.max_query_len is not None:
            max_query_len = self.max_query_len  # dataset-wide Nmax
        else:
            max_query_len = max(queries_lengths)

        if self.max_context_len is not None:
            max_context_len = self.max_context_len  # dataset-wide Nmax
        else:
            max_context_len = max(contexts_lengths)

        batch_size = len(queries)
        dtype = queries[0].dtype
        device = queries[0].device

        # Fill everything with EOS = (0, 0, 0, 0, 1)
        eos_row = torch.tensor([0, 0, 0, 0, 0, 0, 1], dtype=dtype, device=device)
        context_padding = torch.tensor(
            [0, 0, 0, 0, 0, 0, 0], dtype=dtype, device=device
        )
        padded_query = (
            eos_row.view(1, 1, 7).expand(batch_size, max_query_len, 7).clone()
        )
        padded_context = (
            context_padding.view(1, 1, 7).expand(batch_size, max_context_len, 7).clone()
        )

        for idx, (seq, length) in enumerate(zip(queries, queries_lengths)):
            L = min(length, max_query_len)
            padded_query[idx, :L] = seq[:L]

        for idx, (seq, length) in enumerate(zip(contexts, contexts_lengths)):
            L = min(length, max_context_len)
            padded_context[idx, :L] = seq[:L]

        return {
            "queries": padded_query,
            "contexts": padded_context,
            "queries_lengths": torch.tensor(
                queries_lengths, dtype=torch.long, device=device
            ),
            "contexts_lengths": torch.tensor(
                contexts_lengths, dtype=torch.long, device=device
            ),
        }

    # --- Query extraction: kept exactly as you already had it ---
    def _extract_context_query(
        self,
        tokens: torch.Tensor,
    ) -> Optional[torch.Tensor]:

        reset_idx = self._first_index(tokens[:, 5])
        if reset_idx is None:
            raise ValueError("No reset token found in episode.")
        start_idx = reset_idx + 1
        stop_idx = self._first_index(tokens[:, 6], min_index=start_idx)
        if stop_idx is None or stop_idx <= start_idx + 1:
            raise ValueError("No valid query sketch found in episode.")
        return tokens[:reset_idx].clone(), tokens[start_idx + 1 : stop_idx].clone()

    @staticmethod
    def _first_index(column: torch.Tensor, min_index: int = -1) -> Optional[int]:
        indices = torch.nonzero(column > 0.5, as_tuple=False)
        for idx in indices:
            value = int(idx.item())
            if value > min_index:
                return value
        return None

    # --- This is the core: convert your tokens into SketchRNN strokes ---
    # def _tokens_to_sketchrnn_strokes(self, tokens: torch.Tensor) -> torch.Tensor:
    #     """
    #     Convert a (T, D) token tensor into a (T+1, 5) SketchRNN stroke sequence.

    #     Steps:
    #         1. Take coords (x, y).
    #         2. Convert to deltas (Δx, Δy) if needed.
    #         3. Infer stroke boundaries from pen column.
    #         4. Build (Δx, Δy, p1, p2, p3=0) for all real points.
    #         5. Append a single EOS row (0, 0, 0, 0, 1).
    #     """
    #     # 1) coords
    #     deltas = tokens[:, :2].to(dtype=torch.float32)

    #     # 3) infer stroke endings from pen column
    #     pen = tokens[:, 2].to(dtype=torch.float32)  # 1 while drawing, 0 otherwise

    #     pen_down = pen
    #     pen_up = 1.0 - pen_down

    #     # p3 = EOS; only used for explicit EOS + padding
    #     eos = torch.zeros_like(pen_down)

    #     strokes = torch.stack(
    #         [deltas[:, 0], deltas[:, 1], pen_down, pen_up, eos],
    #         dim=-1,
    #     )

    #     # 5) append explicit EOS row (0, 0, 0, 0, 1)
    #     eos_row = self._eos_row(device=strokes.device, dtype=strokes.dtype)
    #     strokes = torch.cat([strokes, eos_row], dim=0)

    #     return strokes

    # def _truncate_with_eos(self, strokes: torch.Tensor, max_len: int) -> torch.Tensor:
    #     """
    #     Truncate to max_len but ensure last row is EOS (0, 0, 0, 0, 1).

    #     This is important to keep exactly one explicit EOS in truncated sequences.
    #     """
    #     truncated = strokes[:max_len].clone()
    #     # overwrite last row with a clean EOS row
    #     truncated[-1] = 0.0
    #     truncated[-1, -1] = 1.0
    #     return truncated

    @staticmethod
    def _eos_row(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        row = torch.zeros(1, 5, device=device, dtype=dtype)
        row[0, -1] = 1.0
        return row
