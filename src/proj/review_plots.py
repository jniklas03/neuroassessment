"""Review figures: subject distributions, paired conditions, and trajectories."""

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from io import BytesIO
import numpy as np

from .motor_metrics import PRIMARY_METRICS
from .session import EXPERIMENT_TYPES

GROUP_COLORS = {"control": "#2563eb", "disease": "#e76f51"}
METRIC_LABELS = {
    "tip_dispersion_rms": "Index fingertip variability",
    "pinch_sd": "Pinch variability",
    "pinch_mean": "Mean pinch distance",
    "tip_speed_mean": "Index fingertip speed",
    "tip_path_length": "Index fingertip path length",
    "task_duration_seconds": "Active task duration",
    "time_to_key_seconds": "Time to key insertion",
    "post_key_duration_seconds": "Time after key insertion",
    "recording_duration_seconds": "Whole trial duration",
    "unlock_duration_seconds": "Unlocking attempt duration",
    "initial_rest_tip_dispersion_rms": "Initial rest fingertip variability",
    "final_rest_tip_dispersion_rms": "Final rest fingertip variability",
}


def _unit(metric):
    if metric.endswith("_seconds"):
        return "s"
    if metric == "tip_speed_mean":
        return "palm lengths/s"
    return "palm lengths"


def plot_group_distributions(subjects, metrics=PRIMARY_METRICS):
    fig = Figure(figsize=(16, 3.5 * len(metrics)))
    axes = fig.subplots(len(metrics), 4, squeeze=False)
    rng = np.random.default_rng(42)
    for row, metric in enumerate(metrics):
        for column, experiment in enumerate(EXPERIMENT_TYPES):
            ax = axes[row, column]
            for x, group in enumerate(("control", "disease")):
                values = subjects.loc[(subjects.experiment_type == experiment) & (subjects.group == group), metric].dropna().to_numpy()
                if len(values):
                    ax.scatter(x + rng.uniform(-.09, .09, len(values)), values,
                               color=GROUP_COLORS[group], alpha=.75, s=40)
                    ax.plot([x - .15, x + .15], [values.mean()] * 2, color="black", lw=2)
            ax.set_xticks([0, 1], ["Control", "Disease"])
            ax.set_title(experiment)
            ax.set_ylabel(METRIC_LABELS.get(metric, metric) + f" ({_unit(metric)})")
            ax.grid(axis="y", alpha=.2)
    fig.suptitle("Each dot is one subject; black lines show group means", y=1.01)
    fig.tight_layout()
    return fig


def plot_paired_conditions(subjects, metric="tip_dispersion_rms"):
    fig = Figure(figsize=(12, 4))
    axes = fig.subplots(1, 2)
    for ax, group in zip(axes, ("control", "disease")):
        wide = subjects.loc[subjects.group == group].pivot(index="subject_name", columns="experiment_type", values=metric)
        for _, row in wide.iterrows():
            values = [row.get(experiment, np.nan) for experiment in EXPERIMENT_TYPES]
            ax.plot(range(4), values, "o-", alpha=.5, color=GROUP_COLORS[group])
        ax.set_xticks(range(4), EXPERIMENT_TYPES, rotation=20)
        ax.set_title(group.capitalize())
        ax.set_ylabel(METRIC_LABELS.get(metric, metric) + f" ({_unit(metric)})")
        ax.grid(alpha=.2)
    fig.suptitle("Within-subject conditions; each line is one subject")
    fig.tight_layout()
    return fig


def plot_trial_series(series, title="Selected trial", *, key_in_lock_seconds=None):
    fig = Figure(figsize=(15, 4))
    axes = fig.subplots(1, 3)
    axes[0].plot(series.relative_tip_x, series.relative_tip_y, lw=1, alpha=.7)
    axes[0].set(xlabel="Index tip relative x (palm lengths)", ylabel="Relative y (palm lengths)", title="Wrist-relative trajectory")
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[0].invert_yaxis()
    axes[1].plot(series.time_seconds, series.pinch)
    axes[1].set(xlabel="Recording time (s)", ylabel="Thumb–index distance (palm lengths)", title="Pinch distance")
    axes[2].plot(series.time_seconds, series.tip_speed)
    axes[2].set(xlabel="Recording time (s)", ylabel="Relative speed (palm lengths/s)", title="Index fingertip speed")
    if key_in_lock_seconds is not None:
        for ax in axes[1:]:
            ax.axvline(key_in_lock_seconds, color="#d97706", linestyle="--", label="Key in lock")
            ax.legend()
        nearest = (series.time_seconds - key_in_lock_seconds).abs().idxmin()
        if abs(series.loc[nearest, "time_seconds"] - key_in_lock_seconds) <= .25:
            axes[0].scatter(series.loc[nearest, "relative_tip_x"], series.loc[nearest, "relative_tip_y"],
                            marker="*", s=150, color="#d97706", label="Key in lock", zorder=5)
            axes[0].legend()
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def figure_png(figure):
    """Serialize a figure without publishing any notebook output."""
    buffer = BytesIO()
    try:
        figure.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(figure)


def display_figure_once(figure):
    """Publish exactly one PNG for standalone statistics cells."""
    from IPython.display import Image, display
    display(Image(data=figure_png(figure)))
