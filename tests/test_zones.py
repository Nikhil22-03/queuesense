"""Geometry tests. No model and no video file."""

import pytest

from queuesense.zones import (
    ZoneError,
    assign_zone,
    click_to_normalized,
    denormalize_points,
    ground_contact_point,
    load_zone_file,
    normalize_points,
    point_in_polygon,
    validate_polygon,
    validate_zones,
    zone_for_box,
    zone_warnings,
)


SQUARE = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]


def test_normalize_roundtrip_is_stable():
    pixels = [(320.0, 180.0), (0.0, 0.0), (640.0, 360.0)]
    normalized = normalize_points(pixels, 640, 360)
    assert normalized[0] == pytest.approx((0.5, 0.5))
    restored = denormalize_points(normalized, 640, 360)
    for got, expected in zip(restored, pixels, strict=True):
        assert got == pytest.approx(expected)


def test_click_on_a_resized_preview_uses_reported_size():
    # A 200x100 preview of any source frame. Center click stays the center.
    assert click_to_normalized(100, 50, 200, 100) == pytest.approx((0.5, 0.5))
    assert click_to_normalized(0, 0, 200, 100) == pytest.approx((0.0, 0.0))
    assert click_to_normalized(200, 100, 200, 100) == pytest.approx((1.0, 1.0))


def test_click_outside_preview_is_rejected():
    with pytest.raises(ZoneError):
        click_to_normalized(500, 10, 200, 100)


def test_boundary_points_belong_to_the_polygon():
    assert point_in_polygon((0.0, 5.0), SQUARE)
    assert point_in_polygon((10.0, 10.0), SQUARE)
    assert point_in_polygon((5.0, 5.0), SQUARE)
    assert not point_in_polygon((11.0, 5.0), SQUARE)
    assert not point_in_polygon((-0.1, 5.0), SQUARE)


def test_ground_contact_is_bottom_center():
    assert ground_contact_point((0, 0, 10, 20)) == pytest.approx((5.0, 20.0))


def test_service_zone_wins_when_polygons_overlap():
    queue = [(0.0, 0.0), (70.0, 0.0), (70.0, 100.0), (0.0, 100.0)]
    service = [(50.0, 0.0), (100.0, 0.0), (100.0, 100.0), (50.0, 100.0)]
    both = (60.0, 90.0)
    queue_only = (20.0, 90.0)
    assert assign_zone(both, queue, service) == "service"
    assert assign_zone(queue_only, queue, service) == "queue"
    assert zone_for_box((50.0, 40.0, 70.0, 90.0), queue, service) == "service"
    assert zone_for_box((10.0, 40.0, 30.0, 90.0), queue, service) == "queue"


def test_zero_area_and_self_intersection_are_rejected():
    with pytest.raises(ZoneError, match="zero area"):
        validate_polygon([(0.1, 0.1), (0.2, 0.2), (0.3, 0.3)], "queue")
    with pytest.raises(ZoneError, match="crosses itself"):
        validate_polygon([(0.1, 0.1), (0.9, 0.9), (0.1, 0.9), (0.9, 0.1)], "service")
    with pytest.raises(ZoneError, match="at least 3"):
        validate_polygon([(0.1, 0.1), (0.2, 0.2)], "queue")


def test_pixel_coordinates_are_not_accepted_as_normalized():
    with pytest.raises(ZoneError, match="normalized"):
        validate_polygon([(10, 10), (40, 10), (40, 40)], "queue")


def test_tiny_pixel_area_is_rejected():
    # Normalized area is nonzero, but on a 100x100 frame the shape is under 4 px².
    tiny = [(0.0, 0.0), (0.02, 0.0), (0.0, 0.02)]
    with pytest.raises(ZoneError, match="square pixels"):
        validate_zones(tiny, [(0.5, 0.5), (0.8, 0.5), (0.8, 0.8)], 100, 100)


def test_full_cover_warning():
    queue = [(0.2, 0.2), (0.4, 0.2), (0.4, 0.4)]
    service = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    warnings = zone_warnings(queue, service, 100, 100)
    assert warnings and "priority" in warnings[0]


def test_service_zone_can_be_omitted_when_no_counter_is_visible():
    queue = [(0.1, 0.1), (0.4, 0.1), (0.4, 0.6)]
    cleaned_queue, service = validate_zones(queue, None, 200, 200)
    assert len(cleaned_queue) == 3
    assert service == []
    assert assign_zone((20, 20), denormalize_points(cleaned_queue, 200, 200), service) == "queue"


def test_sample_zone_file_loads(tmp_path=None):
    from queuesense.config import project_root

    queue, service = load_zone_file(project_root() / "data" / "sample_zones.json")
    assert len(queue) >= 3
    assert len(service) >= 3
