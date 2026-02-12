from __future__ import annotations

import base64
from pathlib import Path


def _b64_image(path: Path) -> str | None:
    if not path.exists():
        return None
    data = path.read_bytes()
    return base64.b64encode(data).decode("utf-8")


def inject_global_styles() -> None:
    """
    Minimal, clean styling (Inter-like) using system fonts fallback.
    """
    import streamlit as st

    st.markdown(
        """
        <style>
          /* Use modern product fonts (no AI vibe) */
          html, body, [class*="css"]  {
            font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Inter, Arial, "Noto Sans", "Apple Color Emoji", "Segoe UI Emoji";
          }

          /* Tighten spacing a bit */
          .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

          /* Card style */
          .card {
            border: 1px solid rgba(0,0,0,0.08);
            border-radius: 16px;
            padding: 14px 16px;
            margin-bottom: 12px;
            background: white;
          }
          .muted { color: rgba(0,0,0,0.6); font-size: 0.92rem; }
          .big { font-size: 1.12rem; font-weight: 650; }

          /* Tag pills */
          .pill {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 0.85rem;
            border: 1px solid rgba(0,0,0,0.12);
            margin-right: 6px;
          }

          /* Emphasis for edge tiers */
          .tierA { border-color: rgba(34,197,94,0.35); background: rgba(34,197,94,0.06); }
          .tierB { border-color: rgba(234,179,8,0.35); background: rgba(234,179,8,0.06); }
          .tierC { border-color: rgba(59,130,246,0.35); background: rgba(59,130,246,0.06); }
          .tierD { border-color: rgba(148,163,184,0.35); background: rgba(148,163,184,0.06); }

          /* Make streamlit metric labels slightly cleaner */
          [data-testid="stMetricLabel"] { font-size: 0.9rem; color: rgba(0,0,0,0.65); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def header_with_logo(title: str, subtitle: str) -> None:
    """
    Optional local logo on the left + title.
    Looks clean even if logo missing.
    """
    import streamlit as st

    root = Path(__file__).resolve().parents[2]  # repo root
    assets = root / "assets"
    logo_path = None

    # Prefer nba_wordmark.png, else basketball.png
    if (assets / "nba_wordmark.png").exists():
        logo_path = assets / "nba_wordmark.png"
    elif (assets / "basketball.png").exists():
        logo_path = assets / "basketball.png"

    logo_b64 = _b64_image(logo_path) if logo_path else None

    if logo_b64:
        st.markdown(
            f"""
            <div style="display:flex; align-items:center; gap:14px; margin-bottom:10px;">
              <img src="data:image/png;base64,{logo_b64}" style="height:44px; width:auto; border-radius:10px;" />
              <div>
                <div style="font-size:1.55rem; font-weight:750; line-height:1.1;">{title}</div>
                <div class="muted">{subtitle}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(f"## {title}")
        st.caption(subtitle)
