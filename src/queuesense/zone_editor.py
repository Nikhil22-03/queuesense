"""Click-to-draw queue and service polygons on the first video frame.

streamlit-image-coordinates supplies the click. Points are stored normalized,
using the width and height reported with the click, so a resized preview does
not change the zone.
"""

from __future__ import annotations

import inspect
import json

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from queuesense.visualization import draw_zones
from queuesense.zones import (
    ZoneError,
    click_to_normalized,
    denormalize_points,
    validate_polygon,
    validate_zones,
    zone_warnings,
)

_MAX_PREVIEW_WIDTH = 960


def _resize(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    if width <= _MAX_PREVIEW_WIDTH:
        return frame
    scale = _MAX_PREVIEW_WIDTH / float(width)
    return cv2.resize(frame, (_MAX_PREVIEW_WIDTH, max(1, int(round(height * scale)))), interpolation=cv2.INTER_AREA)


def _preview(frame: np.ndarray, polygons: dict) -> np.ndarray:
    height, width = frame.shape[:2]
    queue = polygons["queue"] if len(polygons["queue"]) >= 3 else []
    service = polygons["service"] if len(polygons["service"]) >= 3 else []
    canvas = draw_zones(
        frame,
        denormalize_points(queue, width, height) if queue else [],
        denormalize_points(service, width, height) if service else [],
    )
    colors = {"queue": (60, 170, 60), "service": (0, 140, 255)}
    for name, color in colors.items():
        for point in polygons[name]:
            px, py = denormalize_points([tuple(point)], width, height)[0]
            cv2.circle(canvas, (int(round(px)), int(round(py))), 5, color, -1)
    return _resize(canvas)


def _remember_click() -> None:
    payload = st.session_state.get("zone_click")
    if not payload or "x" not in payload:
        return
    stamp = payload.get("unix_time")
    if stamp is not None and stamp == st.session_state.get("last_click_stamp"):
        return
    width = payload.get("width") or st.session_state.get("zone_preview_width")
    height = payload.get("height") or st.session_state.get("zone_preview_height")
    try:
        point = click_to_normalized(payload["x"], payload["y"], width, height)
    except (ZoneError, TypeError, ZeroDivisionError):
        return
    if stamp is not None:
        st.session_state.last_click_stamp = stamp
    target = st.session_state.get("active_zone", "queue")
    points = st.session_state.polygons[target]
    if points and abs(points[-1][0] - point[0]) < 1e-4 and abs(points[-1][1] - point[1]) < 1e-4:
        return
    points.append([point[0], point[1]])
    st.session_state.zones_ready = False
    st.session_state.zone_error = None


def edit_zones(frame: np.ndarray, video_width: int, video_height: int) -> None:
    """Draw, undo, reset, and save the two polygons into session state."""
    st.subheader("Zones")
    st.caption(
        "Queue is green. Service is orange. Click vertices in order around each region. "
        "The service zone wins where the two shapes overlap. "
        "Saved points are fractions of the frame width and height."
    )
    try:
        from streamlit_image_coordinates import streamlit_image_coordinates
    except ImportError:
        st.error("Install streamlit-image-coordinates to draw zones. It is listed in requirements.txt.")
        return

    if "active_zone" not in st.session_state:
        st.session_state.active_zone = "queue"
    st.radio(
        "Polygon being edited",
        options=["queue", "service"],
        format_func=lambda name: "Queue zone" if name == "queue" else "Service zone",
        horizontal=True,
        key="active_zone",
    )
    preview = _preview(frame, st.session_state.polygons)
    st.session_state.zone_preview_width = int(preview.shape[1])
    st.session_state.zone_preview_height = int(preview.shape[0])
    rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    click_kwargs = {"key": "zone_click", "cursor": "crosshair"}
    parameters = inspect.signature(streamlit_image_coordinates).parameters
    if "cursor" not in parameters:
        click_kwargs.pop("cursor")
    if "on_click" in parameters:
        click_kwargs["on_click"] = _remember_click
        streamlit_image_coordinates(image, **click_kwargs)
    else:
        value = streamlit_image_coordinates(image, **click_kwargs)
        if value:
            st.session_state.zone_click = value
            _remember_click()

    queue_count = len(st.session_state.polygons["queue"])
    service_count = len(st.session_state.polygons["service"])
    st.write(f"Queue points: {queue_count}. Service points: {service_count}. Each zone needs at least 3.")
    undo, reset_one, reset_both, save = st.columns(4)
    target = st.session_state.active_zone
    if undo.button("Undo point"):
        if st.session_state.polygons[target]:
            st.session_state.polygons[target].pop()
        st.session_state.zones_ready = False
        st.rerun()
    if reset_one.button("Reset this zone"):
        st.session_state.polygons[target] = []
        st.session_state.zones_ready = False
        st.rerun()
    if reset_both.button("Reset both"):
        st.session_state.polygons = {"queue": [], "service": []}
        st.session_state.zones_ready = False
        st.rerun()
    if save.button("Save zones"):
        try:
            queue, service = validate_zones(
                st.session_state.polygons["queue"],
                st.session_state.polygons["service"],
                video_width,
                video_height,
            )
        except ZoneError as exc:
            st.session_state.zones_ready = False
            st.session_state.zone_error = str(exc)
        else:
            st.session_state.polygons = {
                "queue": [[x, y] for x, y in queue],
                "service": [[x, y] for x, y in service],
            }
            st.session_state.zones_ready = True
            st.session_state.zone_error = None
            st.session_state.zone_warnings = zone_warnings(queue, service, video_width, video_height)

    if st.session_state.get("zone_error"):
        st.error(st.session_state.zone_error)
    elif st.session_state.get("zones_ready"):
        st.success("Zones saved for the next analysis.")
    for warning in st.session_state.get("zone_warnings") or []:
        if st.session_state.get("zones_ready"):
            st.warning(warning)

    uploaded = st.file_uploader("Or load a normalized zone JSON file", type=["json"], key="zone_json_upload")
    if uploaded is not None:
        digest = hash(uploaded.getvalue())
        if digest != st.session_state.get("zone_json_digest"):
            st.session_state.zone_json_digest = digest
            try:
                payload = json.loads(uploaded.getvalue().decode("utf-8"))
                queue = validate_polygon(payload["queue"], "queue")
                service = validate_polygon(payload["service"], "service")
                validate_zones(queue, service, video_width, video_height)
            except (KeyError, TypeError, json.JSONDecodeError, ZoneError) as exc:
                st.session_state.zones_ready = False
                st.session_state.zone_error = f"Could not load that zone file: {exc}"
            else:
                st.session_state.polygons = {
                    "queue": [[x, y] for x, y in queue],
                    "service": [[x, y] for x, y in service],
                }
                st.session_state.zones_ready = True
                st.session_state.zone_error = None
                st.session_state.zone_warnings = zone_warnings(queue, service, video_width, video_height)
                st.rerun()

    if st.session_state.get("zones_ready"):
        payload = {
            "coordinate_space": "normalized",
            "queue": st.session_state.polygons["queue"],
            "service": st.session_state.polygons["service"],
        }
        st.download_button(
            "Download zones JSON",
            data=json.dumps(payload, indent=2),
            file_name="zones.json",
            mime="application/json",
        )
