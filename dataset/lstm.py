"""
Utilities for collating QuickDraw episodes into SketchRNN-friendly batches.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

__all__ = ["InContextSketchRNNCollator", "ILRNNCollator"]


class InContextSketchRNNCollator:
    """
    TODO: Docstring for InContextSketchRNNCollator.
    """

    def __init__(
        self,
        *,
        max_query_len: Optional[int],
        max_context_len: Optional[int],
        teacher_forcing_with_eos: bool = False,
    ) -> None:
        if max_query_len is not None and max_query_len < 2:
            raise ValueError("max_query_len must be >= 2 when provided.")
        if max_context_len is not None and max_context_len < 2:
            raise ValueError("max_context_len must be >= 2 when provided.")
        self.max_query_len = max_query_len
        self.max_context_len = max_context_len
        self.teacher_forcing_with_eos = teacher_forcing_with_eos

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        queries: List[torch.Tensor] = []
        queries_lengths: List[int] = []
        contexts: List[torch.Tensor] = []
        contexts_lengths: List[int] = []

        for sample in batch:
            tokens = sample["tokens"]
            context, query = self._extract_context_query(tokens)

            queries.append(query)
            queries_lengths.append(
                query.shape[0]
                if query.shape[0] <= self.max_query_len or self.max_query_len is None
                else self.max_query_len
            )

            contexts.append(context)
            contexts_lengths.append(
                context.shape[0]
                if context.shape[0] <= self.max_context_len
                or self.max_context_len is None
                else self.max_context_len
            )

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

        if self.teacher_forcing_with_eos:
            queries_lengths = [max_query_len for _ in queries_lengths]
        return {
            "queries": padded_query[:, :, [0, 1, 2, 3, 4, 6]],
            "contexts": padded_context,
            "queries_lengths": torch.tensor(
                queries_lengths, dtype=torch.long, device=device
            ),
            "contexts_lengths": torch.tensor(
                contexts_lengths, dtype=torch.long, device=device
            ),
        }

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
        return tokens[: reset_idx + 1].clone(), tokens[start_idx:].clone()

    @staticmethod
    def _first_index(column: torch.Tensor, min_index: int = -1) -> Optional[int]:
        indices = torch.nonzero(column > 0.5, as_tuple=False)
        for idx in indices:
            value = int(idx.item())
            if value > min_index:
                return value
        return None

    @staticmethod
    def _eos_row(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        row = torch.zeros(1, 5, device=device, dtype=dtype)
        row[0, -1] = 1.0
        return row


class ILRNNCollator:
    """
    TODO: Docstring for InContextSketchRNNCollator.
    """

    def __init__(
        self,
        *,
        max_query_len: Optional[int],
        teacher_forcing_with_eos: bool = False,
    ) -> None:
        if max_query_len is not None and max_query_len < 2:
            raise ValueError("max_query_len must be >= 2 when provided.")
        self.max_query_len = max_query_len
        self.teacher_forcing_with_eos = teacher_forcing_with_eos

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        output_queries: List[torch.Tensor] = []
        input_queries: List[torch.Tensor] = []
        queries_lengths: List[int] = []

        for sample in batch:
            tokens = sample["tokens"]
            _, query = self._extract_context_query(tokens)

            output_queries.append(query)
            input_queries.append(query)
            queries_lengths.append(
                query.shape[0]
                if query.shape[0] <= self.max_query_len or self.max_query_len is None
                else self.max_query_len
            )

        if not output_queries:
            raise ValueError(
                "No valid query sketches found in batch for SketchRNNCollator."
            )

        if self.max_query_len is not None:
            max_query_len = self.max_query_len
        else:
            max_query_len = max(queries_lengths)

        batch_size = len(output_queries)
        dtype = output_queries[0].dtype
        device = output_queries[0].device

        eos_row = torch.tensor([0, 0, 0, 0, 0, 0, 1], dtype=dtype, device=device)
        context_padding = torch.tensor(
            [0, 0, 0, 0, 0, 0, 0], dtype=dtype, device=device
        )
        padded_output_query = (
            eos_row.view(1, 1, 7).expand(batch_size, max_query_len, 7).clone()
        )
        padded_input_query = (
            context_padding.view(1, 1, 7).expand(batch_size, max_query_len, 7).clone()
        )

        for idx, (seq, length) in enumerate(zip(output_queries, queries_lengths)):
            L = min(length, max_query_len)
            padded_output_query[idx, :L] = seq[:L]

        for idx, (seq, length) in enumerate(zip(input_queries, queries_lengths)):
            L = min(length, max_query_len)
            padded_input_query[idx, :L] = seq[:L]

        input_queries_lengths = queries_lengths.copy()
        if self.teacher_forcing_with_eos:
            queries_lengths = [max_query_len for _ in queries_lengths]

        return {
            "output_queries": padded_output_query[:, :, [0, 1, 2, 3, 4, 6]],
            "input_queries": padded_input_query,
            "output_queries_lengths": torch.tensor(
                queries_lengths, dtype=torch.long, device=device
            ),
            "input_queries_lengths": torch.tensor(
                input_queries_lengths, dtype=torch.long, device=device
            ),
        }

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
        return tokens[: reset_idx + 1].clone(), tokens[start_idx:].clone()

    @staticmethod
    def _first_index(column: torch.Tensor, min_index: int = -1) -> Optional[int]:
        indices = torch.nonzero(column > 0.5, as_tuple=False)
        for idx in indices:
            value = int(idx.item())
            if value > min_index:
                return value
        return None

    @staticmethod
    def _eos_row(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        row = torch.zeros(1, 5, device=device, dtype=dtype)
        row[0, -1] = 1.0
        return row
