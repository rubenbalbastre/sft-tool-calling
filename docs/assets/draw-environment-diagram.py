#!/usr/bin/env python3
"""Draw the project environment overview as an editable SVG."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "docs" / "assets" / "environment-overview.svg"


def box(x, y, width, height, title, subtitle="", accent="#2563eb"):
    subtitle_svg = ""
    if subtitle:
        lines = subtitle.split("\n")
        subtitle_svg = "".join(
            f'<text x="{x + width / 2}" y="{y + 58 + index * 18}" '
            f'class="body" text-anchor="middle">{line}</text>'
            for index, line in enumerate(lines)
        )
    return f"""
    <rect x="{x}" y="{y}" width="{width}" height="{height}" rx="12"
          fill="#ffffff" stroke="#cbd5e1" stroke-width="2"/>
    <rect x="{x}" y="{y}" width="6" height="{height}" rx="3" fill="{accent}"/>
    <text x="{x + width / 2}" y="{y + 33}" class="title" text-anchor="middle">{title}</text>
    {subtitle_svg}
    """


def arrow(x1, y1, x2, y2, label, color="#2563eb"):
    middle_x = (x1 + x2) / 2
    return f"""
    <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"
          stroke="{color}" stroke-width="2.5" marker-end="url(#arrow-{color[1:]})"/>
    <text x="{middle_x}" y="{y1 - 10}" class="label" text-anchor="middle">{label}</text>
    """


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1320" height="510"
         viewBox="0 0 1320 510" role="img"
         aria-labelledby="diagram-title diagram-description">
  <title id="diagram-title">Procurement tool-calling environment</title>
  <desc id="diagram-description">A model interacts with procurement tools backed by seeded SQLite data, then an executable verifier returns success and reward.</desc>
  <defs>
    <marker id="arrow-2563eb" markerWidth="8" markerHeight="8" refX="7" refY="3"
            orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L0,6 L9,3 z" fill="#2563eb"/>
    </marker>
    <marker id="arrow-0f766e" markerWidth="8" markerHeight="8" refX="7" refY="3"
            orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L0,6 L9,3 z" fill="#0f766e"/>
    </marker>
    <marker id="arrow-d97706" markerWidth="8" markerHeight="8" refX="7" refY="3"
            orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L0,6 L9,3 z" fill="#d97706"/>
    </marker>
    <style>
      text {{ font-family: DejaVu Sans, Arial, sans-serif; }}
      .heading {{ font-size: 25px; font-weight: 700; fill: #0f172a; }}
      .title {{ font-size: 18px; font-weight: 700; fill: #0f172a; }}
      .body {{ font-size: 15px; font-weight: 400; fill: #475569; }}
      .label {{ font-size: 13px; font-weight: 600; fill: #334155; }}
      .small {{ font-size: 12px; font-weight: 600; fill: #475569; }}
    </style>
  </defs>
  <rect width="1320" height="510" fill="#f8fafc"/>
  <text x="60" y="52" class="heading">Executable procurement environment</text>
  <text x="60" y="79" class="body">Different prompts require different tool routes; success is checked from outcomes, not an exact trace.</text>

  {box(50, 130, 220, 160, "User request", "Constraints\nPreferences", "#7c3aed")}
  {box(360, 130, 260, 160, "Tool-calling model", "Chooses the next tool\nUses observations", "#2563eb")}
  {box(710, 100, 360, 220, "Procurement tools", "search_suppliers\nget_supplier_profile\nrequest_quote\nget_delivery_options\nsubmit_procurement_plan\nreport_no_feasible_option", "#0f766e")}
  {box(1150, 130, 150, 160, "Verifier", "Evidence\nFeasibility\nUtility", "#d97706")}

  {arrow(270, 210, 360, 210, "prompt", "#2563eb")}
  {arrow(620, 180, 710, 180, "tool call", "#0f766e")}
  {arrow(710, 245, 620, 245, "observation", "#d97706")}
  {arrow(1070, 210, 1150, 210, "final action", "#2563eb")}

  <rect x="805" y="350" width="170" height="74" rx="12" fill="#eff6ff"
        stroke="#93c5fd" stroke-width="2"/>
  <ellipse cx="890" cy="370" rx="55" ry="12" fill="#bfdbfe" stroke="#2563eb"/>
  <path d="M835 370 v30 c0 16 110 16 110 0 v-30" fill="#dbeafe" stroke="#2563eb"/>
  <text x="890" y="398" class="title" text-anchor="middle">SQLite</text>
  <text x="890" y="444" class="small" text-anchor="middle">Fixed master data + seeded episode state</text>
  <line x1="890" y1="350" x2="890" y2="320" stroke="#2563eb" stroke-width="2.5"
        marker-end="url(#arrow-2563eb)"/>

  <path d="M1225 290 V480 H490 V290" fill="none" stroke="#d97706" stroke-width="2.5"
        marker-end="url(#arrow-d97706)"/>
  <text x="1180" y="465" class="label" text-anchor="end">success + reward</text>

</svg>
"""
    OUTPUT.write_text(svg, encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
