"""Factory functions for supervision annotators, driven by AppConfig."""

import supervision as sv

from football_vision.config import AppConfig


def make_box_annotator(cfg: AppConfig) -> sv.BoxAnnotator:
    return sv.BoxAnnotator(
        color=sv.ColorPalette.from_hex([f"#{c}" for c in cfg.color_box_palette]),
        thickness=2,
    )


def make_label_annotator(cfg: AppConfig, position=sv.Position.BOTTOM_CENTER) -> sv.LabelAnnotator:
    return sv.LabelAnnotator(
        color=sv.ColorPalette.from_hex([f"#{c}" for c in cfg.color_box_palette]),
        text_color=sv.Color.from_hex("#000000"),
        text_position=position,
    )


def make_ellipse_annotator(cfg: AppConfig) -> sv.EllipseAnnotator:
    return sv.EllipseAnnotator(
        color=sv.ColorPalette.from_hex(
            [f"#{cfg.color_team_0}", f"#{cfg.color_team_1}", f"#{cfg.color_referee}"]
        ),
        thickness=cfg.ellipse_thickness,
    )


def make_triangle_annotator(cfg: AppConfig) -> sv.TriangleAnnotator:
    return sv.TriangleAnnotator(
        color=sv.Color.from_hex(f"#{cfg.color_ball}"),
        base=cfg.triangle_base,
        height=cfg.triangle_height,
        outline_thickness=cfg.triangle_outline_thickness,
    )


def make_vertex_annotator(cfg: AppConfig) -> sv.VertexAnnotator:
    return sv.VertexAnnotator(
        color=sv.Color.from_hex(f"#{cfg.color_team_1}"),
        radius=cfg.keypoint_radius,
    )
