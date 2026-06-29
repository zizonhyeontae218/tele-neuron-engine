from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import numpy as np

from tele_neuron.config import ExperimentConfig
from tele_neuron.physics import SimulationState, initialize_state, step_state
from tele_neuron.readout import count_in_zone, readout_bits


def run_display(
    config: ExperimentConfig,
    bits: tuple[int, ...],
    *,
    frames: int | None = None,
    interval_ms: int = 90,
    save_path: str | Path | None = None,
) -> None:
    state = initialize_state(config)
    frame_count = frames or config.simulation.steps

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_xlim(0, config.space.size[0])
    ax.set_ylim(0, config.space.size[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(_title(config, bits, 0, state))

    scatter = ax.scatter(
        state.positions[:, 0],
        state.positions[:, 1],
        c=state.positions[:, 2],
        cmap="viridis",
        vmin=0,
        vmax=config.space.size[2],
        s=max(20, config.balls.radius * 450),
        alpha=0.85,
        edgecolors="black",
        linewidths=0.25,
    )
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("z")

    _draw_input_points(ax, config, bits)
    _draw_readout_zones(ax, config)

    def update(frame: int) -> tuple[object, ...]:
        step_state(state, config, bits)
        scatter.set_offsets(state.positions[:, :2])
        scatter.set_array(state.positions[:, 2])
        ax.set_title(_title(config, bits, frame + 1, state))
        return (scatter,)

    animation = FuncAnimation(
        fig,
        update,
        frames=frame_count,
        interval=interval_ms,
        blit=False,
        repeat=False,
    )

    if save_path is not None:
        output = Path(save_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        animation.save(output, writer=PillowWriter(fps=max(1, 1000 // interval_ms)))
        plt.close(fig)
        return

    plt.show()


def save_final_snapshot(
    config: ExperimentConfig,
    bits: tuple[int, ...],
    path: str | Path,
) -> None:
    state = initialize_state(config)
    for _ in range(config.simulation.steps):
        step_state(state, config, bits)

    fig = Figure(figsize=(8, 7))
    ax = fig.subplots()
    ax.set_xlim(0, config.space.size[0])
    ax.set_ylim(0, config.space.size[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(_title(config, bits, config.simulation.steps, state))
    scatter = ax.scatter(
        state.positions[:, 0],
        state.positions[:, 1],
        c=state.positions[:, 2],
        cmap="viridis",
        vmin=0,
        vmax=config.space.size[2],
        s=max(20, config.balls.radius * 450),
        alpha=0.85,
        edgecolors="black",
        linewidths=0.25,
    )
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("z")
    _draw_input_points(ax, config, bits)
    _draw_readout_zones(ax, config)

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140, bbox_inches="tight")


def _title(
    config: ExperimentConfig,
    bits: tuple[int, ...],
    step: int,
    state: SimulationState,
) -> str:
    counts = [count_in_zone(state.positions, zone) for zone in config.readout]
    output = readout_bits(state.positions, config)
    bit_text = "".join(str(bit) for bit in bits)
    return (
        f"{config.name} | bits={bit_text} | step={step}/{config.simulation.steps} | "
        f"readout_count={counts} | output={output}"
    )


def _draw_input_points(
    ax: plt.Axes,
    config: ExperimentConfig,
    bits: tuple[int, ...],
) -> None:
    points = np.array(config.input.points, dtype=np.float64)
    for index, point in enumerate(points):
        is_active = index < len(bits) and bits[index] == 1
        ax.scatter(
            point[0],
            point[1],
            marker="X",
            s=130,
            color="crimson" if is_active else "silver",
            edgecolors="black",
            linewidths=0.75,
            zorder=4,
        )
        ax.text(point[0] + 0.06, point[1] + 0.06, f"in{index}", fontsize=9)


def _draw_readout_zones(ax: plt.Axes, config: ExperimentConfig) -> None:
    for zone in config.readout:
        x = zone.center[0] - zone.half_extents[0]
        y = zone.center[1] - zone.half_extents[1]
        width = zone.half_extents[0] * 2
        height = zone.half_extents[1] * 2
        ax.add_patch(
            Rectangle(
                (x, y),
                width,
                height,
                fill=False,
                edgecolor="darkorange",
                linewidth=2,
                linestyle="--",
            )
        )
        ax.text(x, y + height + 0.06, zone.name, color="darkorange", fontsize=9)
