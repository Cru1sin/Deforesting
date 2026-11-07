"""LSTM, GRU and MLP stopping networks with explicit FiLM and capacity."""

import torch
from torch import nn


class StoppingModel(nn.Module):
    """Shared cycle-wise dropout and FiLM for recurrent or current-only encoders."""

    def __init__(
        self,
        width,
        numeric_width=0,
        *,
        hidden_width=32,
        recurrent_layers=1,
        reset_history=False,
        mlp_layers=2,
        mlp_widths=None,
        cell="GRU",
        visual_skip=False,
        film=False,
        visual_dropout=0.0,
        output_dropout=0.0,
        visual_time_shared=True,
        output_time_shared=False,
    ):
        super().__init__()
        self.numeric_width = numeric_width
        self.reset_history = reset_history
        self.visual_dropout = visual_dropout
        self.output_dropout = nn.Dropout(output_dropout)
        self.visual_time_shared = visual_time_shared
        self.output_time_shared = output_time_shared
        self.cell = cell
        if cell == "MLP":
            widths = tuple(mlp_widths or [hidden_width] * mlp_layers)
            if not widths or any(w < 1 for w in widths):
                raise ValueError("MLP widths must be positive")
            self.mlp = nn.Sequential(
                *[
                    layer
                    for before, after in zip((width, *widths[:-1]), widths)
                    for layer in (nn.Linear(before, after), nn.ReLU())
                ]
            )
            hidden_width = widths[-1]
        else:
            recurrent = {"GRU": nn.GRU, "LSTM": nn.LSTM}[cell]
            # Keep the serialized "gru.*" parameter keys of the frozen weights.
            self.gru = recurrent(
                width,
                hidden_width,
                num_layers=recurrent_layers,
                batch_first=True,
            )
        self.head = nn.Linear(hidden_width, 1)
        self.visual_skip = nn.Linear(width - numeric_width, 1, bias=False) if visual_skip else None
        if self.visual_skip is not None:
            nn.init.zeros_(self.visual_skip.weight)
        self.film = nn.Linear(numeric_width, 2 * hidden_width) if film else None
        if self.film is not None:
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)

    def forward(self, values, lengths=None):
        if self.training and self.visual_dropout:
            # One feature mask per cycle, shared across time; controls stay observed.
            visual = values[..., self.numeric_width :]
            mask = torch.nn.functional.dropout(
                torch.ones_like(visual[:, :1] if self.visual_time_shared else visual),
                p=self.visual_dropout,
                training=True,
            )
            values = torch.cat([values[..., : self.numeric_width], visual * mask], dim=-1)
        recurrent = values
        if self.cell == "MLP":
            hidden = self.mlp(recurrent)
        elif self.reset_history:
            # Independent single-step states with the same recurrent parameters.
            hidden, _ = self.gru(recurrent.reshape(-1, 1, recurrent.shape[-1]))
            hidden = hidden.reshape(*recurrent.shape[:2], -1)
        else:
            hidden, _ = self.gru(recurrent)
        logits = self.readout(hidden, values[..., : self.numeric_width])
        if self.visual_skip is not None:
            logits = logits + self.visual_skip(values[..., self.numeric_width :])
        return logits.squeeze(-1)

    def readout(self, hidden, numeric):
        """Shared actual/probed readout from a recurrent state and current controls."""
        if self.film is not None:
            gamma, beta = self.film(numeric).chunk(2, dim=-1)
            hidden = (1 + gamma) * hidden + beta
        if self.training and self.output_time_shared and self.output_dropout.p:
            hidden = hidden * self.output_dropout(torch.ones_like(hidden[:, :1]))
        else:
            hidden = self.output_dropout(hidden)
        logits = self.head(hidden)
        return logits


def build_model(
    name,
    numeric_width,
    visual_width,
    seed,
    dropout_config=None,
    hidden_width=32,
    recurrent_layers=1,
    reset_history=False,
    mlp_layers=2,
    mlp_widths=None,
    film=None,
):
    torch.manual_seed(seed)
    width = numeric_width + visual_width
    if name in ("LSTM", "GRU", "MLP"):
        if film is None:
            raise ValueError("Specify film=True or film=False for a named network")
        if film and numeric_width == 0:
            raise ValueError("FiLM requires operating-condition inputs")
        return StoppingModel(
            width,
            numeric_width,
            cell=name,
            hidden_width=hidden_width,
            recurrent_layers=recurrent_layers,
            reset_history=reset_history,
            mlp_layers=mlp_layers,
            mlp_widths=mlp_widths,
            film=film,
            **(dropout_config or {}),
        )
    # Frozen experiments encode cell/FiLM/dropout in their architecture names.
    # Keep their original defaults when restoring those saved weights.
    if film is not None:
        raise ValueError("Use LSTM, GRU or MLP when specifying FiLM explicitly")
    if name == "LSTM_plain":
        cell = "LSTM"
        return StoppingModel(
            width,
            numeric_width,
            cell=cell,
            hidden_width=hidden_width,
            recurrent_layers=recurrent_layers,
            reset_history=reset_history,
            **(dropout_config or {}),
        )
    if name in (
        "MLP_plain",
        "MLP_film",
        "LSTM32_film",
        "GRU32",
        "GRU32_skip",
        "GRU32_visualdrop",
        "GRU32_drop",
        "GRU32_film",
        "GRU32_film_skip",
    ):
        dropout = dict(
            visual_dropout=0.1
            if name
            in (
                "MLP_film",
                "LSTM32_film",
                "GRU32_visualdrop",
                "GRU32_film",
                "GRU32_film_skip",
            )
            else 0.0,
            output_dropout=0.2 if name == "GRU32_drop" else 0.0,
        )
        dropout.update(dropout_config or {})
        if name in ("MLP_film", "MLP_plain"):
            cell = "MLP"
        elif name.startswith("LSTM"):
            cell = "LSTM"
        else:
            cell = "GRU"
        return StoppingModel(
            width,
            numeric_width,
            cell=cell,
            hidden_width=hidden_width,
            recurrent_layers=recurrent_layers,
            reset_history=reset_history,
            mlp_layers=mlp_layers,
            mlp_widths=mlp_widths,
            visual_skip=name in ("GRU32_skip", "GRU32_film_skip"),
            film=name in ("MLP_film", "LSTM32_film", "GRU32_film", "GRU32_film_skip"),
            **dropout,
        )
    if (
        name == "logistic"
        and dropout_config
        and not any(dropout_config.get(k, 0.0) for k in ("visual_dropout", "output_dropout"))
    ):
        dropout_config = None
    if dropout_config:
        raise ValueError("Configured dropout requires a recurrent or MLP network")
    if name == "logistic":
        return nn.Sequential(nn.Linear(width, 1), nn.Flatten(0))
    raise ValueError(f"Unknown stopping model: {name}")
