"""Self-contained interactive replay of recorded hand landmarks."""

import numpy as np
import plotly.graph_objects as go
from html import escape
import json

from .review_data import COORDINATES

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


def replay_figure(data, *, title="Landmark replay", fps=15, max_frames=400,
                  max_gap_seconds=0.15, key_in_lock_seconds=None):
    """Resample the recording clock; blank gaps instead of bridging missing hands.

    Estimated depth is wrist-relative, not a calibrated 3D point cloud. Multiple
    hands therefore do not share a physical depth origin. Detection indices in
    legacy data are shown explicitly, rather than assigned anatomical identities.
    """
    if data.empty:
        raise ValueError("No detected landmarks to replay")
    if fps <= 0 or max_frames < 2 or max_gap_seconds <= 0:
        raise ValueError("fps, max_gap_seconds must be positive and max_frames at least two")
    times = data.timestamp_ms.to_numpy(dtype=float) / 1000
    first, last = times.min(), times.max()
    if key_in_lock_seconds is not None:
        if not np.isfinite(key_in_lock_seconds) or key_in_lock_seconds < 0:
            raise ValueError("Key event time must be finite and nonnegative")
        first, last = min(first, key_in_lock_seconds), max(last, key_in_lock_seconds)
    count = min(max_frames, max(2, int(np.ceil((last - first) * fps)) + 1))
    timeline = np.linspace(first, last, count)
    event_frame = None
    if key_in_lock_seconds is not None:
        event_frame = int(np.argmin(abs(timeline - key_in_lock_seconds)))
        timeline[event_frame] = key_in_lock_seconds
    delay_ms = max(1, (last - first) * 1000 / max(1, count - 1))
    labels = set(data.handedness) & {"left", "right"}
    tracks = [(name, data.loc[data.handedness == name]) for name in sorted(labels)]
    unknown = data.loc[~data.handedness.isin(["left", "right"])]
    tracks.extend((f"Detection index {hand} (identity unknown)", rows)
                  for hand, rows in unknown.groupby("hand"))
    palettes = ["#2563eb", "#e76f51", "#10b981", "#9333ea"]
    prepared = []
    for name, rows in tracks:
        # Anatomical duplicates in a frame cannot be assigned unambiguously.
        rows = rows.loc[~rows.duplicated("frame", keep=False)].sort_values("timestamp_ms")
        if not rows.empty:
            prepared.append((name, rows.timestamp_ms.to_numpy() / 1000,
                             rows[COORDINATES].to_numpy().reshape(-1, 21, 3)))
    if not prepared:
        raise ValueError("No unambiguous landmark tracks to replay")

    def traces_at(timestamp):
        traces = []
        for index, (name, track_times, points) in enumerate(prepared):
            insertion = np.searchsorted(track_times, timestamp)
            candidates = [i for i in (insertion - 1, insertion) if 0 <= i < len(track_times)]
            nearest = min(candidates, key=lambda i: abs(track_times[i] - timestamp))
            visible = abs(track_times[nearest] - timestamp) <= max_gap_seconds
            xyz = points[nearest] if visible else np.full((21, 3), np.nan)
            color = palettes[index % len(palettes)]
            traces.append(go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                                      mode="markers", marker=dict(size=6 if timestamp == key_in_lock_seconds else 4,
                                                       color="#d97706" if timestamp == key_in_lock_seconds else color),
                                      text=[f"Landmark {i}" for i in range(21)],
                                      hovertemplate="%{text}<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra>%{fullData.name}</extra>",
                                      name=name, legendgroup=name))
            segments = np.array([xyz[p] for a, b in CONNECTIONS for p in (a, b)])
            line = np.full((len(CONNECTIONS) * 3, 3), np.nan)
            line.reshape(-1, 3, 3)[:, :2] = segments.reshape(-1, 2, 3)
            traces.append(go.Scatter3d(x=line[:, 0], y=line[:, 1], z=line[:, 2],
                                      mode="lines", line=dict(color=color, width=3),
                                      showlegend=False, legendgroup=name, hoverinfo="skip"))
        return traces

    def event_annotation(timestamp):
        if key_in_lock_seconds is None:
            return []
        status = "KEY IN LOCK" if timestamp == key_in_lock_seconds else (
            "Before insertion" if timestamp < key_in_lock_seconds else "After insertion")
        return [dict(text=f"Key in lock at {key_in_lock_seconds:.2f}s · {status}",
                     x=.5, y=1.02, xref="paper", yref="paper", showarrow=False,
                     font=dict(color="#d97706", size=16))]

    frames = [go.Frame(name=str(i), data=traces_at(float(t)),
                       layout=dict(annotations=event_annotation(float(t))))
              for i, t in enumerate(timeline)]
    fig = go.Figure(data=frames[0].data, frames=frames)
    depth = data[[f"z{i}" for i in range(21)]].to_numpy()
    z_min, z_max = min(-.1, float(depth.min())), max(.1, float(depth.max()))
    fig.update_layout(
        annotations=event_annotation(float(timeline[0])),
        title=title + " · estimated wrist-relative depth", height=570,
        paper_bgcolor="#111827", plot_bgcolor="#111827", font=dict(color="#e5e7eb"),
        template="plotly_dark",
        scene=dict(xaxis=dict(title="x / image width", range=[0, 1]),
                   yaxis=dict(title="y / image height", range=[1, 0]),
                   zaxis=dict(title="estimated z / image width", range=[z_min, z_max]),
                   aspectmode="manual", aspectratio=dict(x=1, y=1, z=.4)),
        uirevision="keep-camera", margin=dict(l=0, r=0, b=30, t=70),
        updatemenus=[dict(type="buttons", direction="left", x=0, y=0,
                         buttons=[dict(label="Play", method="animate", args=[None, dict(
                             frame=dict(duration=delay_ms, redraw=True), transition=dict(duration=0), fromcurrent=True)]),
                                  dict(label="Pause", method="animate", args=[[None], dict(
                                      mode="immediate", frame=dict(duration=0, redraw=False), transition=dict(duration=0))])])],
        sliders=[dict(active=0, currentvalue=dict(prefix="Recording time: ", suffix=" s"),
                      steps=[dict(label=f"{t:.2f}" + (" · key" if i == event_frame else ""), method="animate", args=[[str(i)], dict(
                          mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))])
                             for i, t in enumerate(timeline)])],
    )
    if event_frame is not None:
        fig.layout.updatemenus[0].buttons += (dict(
            label="Key in lock", method="animate", args=[[str(event_frame)], dict(
                mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))]),)
    return fig


def replay_html(data, **kwargs):
    """Run the self-contained replay in its own document, outside widget MIME HTML."""
    return _replay_frame_html(replay_figure(data, **kwargs))


def _replay_frame_html(figure):
    document = figure.to_html(
        full_html=True, include_plotlyjs=True, auto_play=False,
        config=dict(responsive=True, displaylogo=False),
    )
    document = document.replace("<head>", '<head><style>html,body{margin:0;background:#111827;} '
                                '.plotly-graph-div{width:100%!important;}</style>', 1)
    return ('<iframe title="Hand landmark replay" sandbox="allow-scripts allow-same-origin" '
            'style="display:block;width:100%;height:590px;border:0;background:#111827" '
            f'srcdoc="{escape(document, quote=True)}"></iframe>')


def display_replay(data, **kwargs):
    """Prefer the notebook's native Plotly renderer, with isolated HTML fallback."""
    from IPython.display import display
    figure = replay_figure(data, **kwargs)
    display({
        "application/vnd.plotly.v1+json": json.loads(figure.to_json()),
        "text/html": _replay_frame_html(figure),
    }, raw=True)


def review_widget(inventory, *, hand_selections=None, allow_legacy=False, legacy_image_size=None,
                  min_confidence=0.6, max_gap_seconds=0.25):
    """Trial selector with replay and a separate task-hand metric preview."""
    import ipywidgets as widgets
    from .motor_metrics import movement_series
    from .review_data import load_metadata, load_trial
    from .review_plots import plot_trial_series, figure_png
    from .task_analysis import load_annotations, save_annotations, key_event_seconds

    valid = inventory.loc[inventory.error == ""]
    if valid.empty:
        return widgets.HTML("No valid recordings available yet.")
    options = [(f"{row.subject_name} | {row.experiment_type} | {row.trial_id}", row.trial_id)
               for _, row in valid.iterrows()]
    dropdown = widgets.Dropdown(options=options, description="Trial:", layout=widgets.Layout(width="95%"))
    button = widgets.Button(description="Load replay", button_style="primary")
    # Fixed widget slots: no kernel Output capture or multi-MIME publication.
    replay_slot = widgets.HTML(layout=widgets.Layout(width="100%"))
    plot_slot = widgets.Image(format="png", layout=widgets.Layout(width="100%"))
    status_slot = widgets.HTML()
    output = widgets.VBox([], layout=widgets.Layout(width="100%"))
    event_time = widgets.FloatText(value=0, description="Key in lock (s):",
                                  style={"description_width": "140px"})
    notes = widgets.Textarea(description="Notes:", layout=widgets.Layout(width="90%"))
    save_button = widgets.Button(description="Save event time", button_style="success")
    annotation_message = widgets.HTML()

    def load_event(_=None):
        annotation_message.value = ""
        # A new selection invalidates the displayed trial until Load is pressed.
        output.children = ()
        replay_slot.value = ""
        plot_slot.value = b""
        try:
            row = valid.loc[valid.trial_id == dropdown.value].iloc[0]
            annotation = load_annotations(row.csv_path) or {}
            event_time.value = key_event_seconds(row.csv_path, load_metadata(row)) or 0
            notes.value = annotation.get("notes", "")
        except (ValueError, OSError) as error:
            annotation_message.value = escape(f"Cannot load event: {error}")

    def save_event(_):
        try:
            row = valid.loc[valid.trial_id == dropdown.value].iloc[0]
            metadata = load_metadata(row)
            annotation = dict(key_in_lock_seconds=event_time.value, notes=notes.value)
            save_annotations(row.csv_path, annotation, float(metadata["duration_seconds"]))
            annotation_message.value = "Saved event. Reload replay and rerun analysis to update results."
        except (ValueError, OSError, KeyError) as error:
            annotation_message.value = escape(f"Cannot save event: {error}")

    dropdown.observe(load_event, names="value")
    save_button.on_click(save_event)
    load_event()

    def show(_):
        button.disabled = True
        output.children = ()
        replay_slot.value = ""
        plot_slot.value = b""
        status_slot.value = ""
        try:
            row = valid.loc[valid.trial_id == dropdown.value].iloc[0]
            data = load_trial(row.csv_path)
            metadata = load_metadata(row)
            insertion = key_event_seconds(row.csv_path, metadata)
            replay_slot.value = replay_html(
                data, title=f"{row.subject_name} · {row.experiment_type}",
                key_in_lock_seconds=insertion,
            )
            try:
                series, hand = movement_series(
                    data, metadata, (hand_selections or {}).get(row.trial_id),
                    allow_legacy=allow_legacy, legacy_image_size=legacy_image_size,
                    min_confidence=min_confidence, max_gap_seconds=max_gap_seconds,
                )
                figure = plot_trial_series(series, title=f"{row.subject_name} · {row.experiment_type} · {hand}",
                                           key_in_lock_seconds=insertion)
                plot_slot.value = figure_png(figure)
                output.children = (replay_slot, plot_slot)
            except ValueError as error:
                status_slot.value = escape(f"Replay available; metric preview needs configuration: {error}")
                output.children = (replay_slot, status_slot)
        except (ValueError, OSError) as error:
            status_slot.value = escape(f"Cannot review this trial: {error}")
            output.children = (status_slot,)
        finally:
            button.disabled = False

    button.on_click(show)
    return widgets.VBox([
        dropdown, button, output,
        widgets.HTML("<b>Key-in-lock event</b> — success is assumed. Times use the recording clock."),
        event_time, notes, save_button, annotation_message,
    ])
