# UI/UX Audit & Fix — Legal RLM Dashboard

**Date:** 2026-04-14
**Branch:** SebihSpecial
**File:** `src/irys/ui/app.py`

---

## Overview

Full UI/UX overhaul of the 6-tab Gradio 6.12.0 legal intelligence dashboard. Goals:

1. Fully responsive layout — no content cutting anywhere
2. Every data type renders in a visually appropriate, properly styled component
3. Matter Intelligence sidebar looks polished and matches the selected Gradio theme (light and dark)
4. Zero functionality removed — all buttons, callbacks, file uploaders, and wiring stay intact

---

## Section 1: Design System Foundation

### Approach

One `_DS_CSS` constant replaces all scattered `<style>` blocks and the existing `_SIDEBAR_STYLE`. Every `_fmt_*` function prepends this constant to its returned HTML. This ensures styles are available at render time regardless of Gradio's asynchronous CSS loading order.

### Theme Compatibility

All colors use Gradio CSS variables so light/dark toggle works automatically:

| Token | CSS Variable | Fallback |
|---|---|---|
| Panel background | `var(--background-fill-secondary)` | `#f8fafc` |
| Card background | `var(--background-fill-primary)` | `#ffffff` |
| Border | `var(--block-border-color)` | `#e2e8f0` |
| Body text | `var(--body-text-color)` | `#1e293b` |
| Muted text | `var(--body-text-color-subdued)` | `#64748b` |
| Accent | `var(--color-accent)` | `#3b82f6` |

Semantic accent colors (green = covered, amber = partial/spend, red = gap) are hardcoded as they are intentional status signals, not theme colors. They remain readable in both light and dark themes.

### Shared Component Classes

All renderers use only these classes — no inline styles:

- `.ds-panel` — outer container (border-radius, padding, border, background)
- `.ds-section-title` — uppercase label, muted color, small font
- `.ds-card` — metric/stat card (border, rounded, padding)
- `.ds-card-value` — large number inside a card
- `.ds-card-label` — small label below the number
- `.ds-bar-row` — label + bar + value laid out horizontally
- `.ds-bar-track` — bar background (muted)
- `.ds-bar-fill` — bar foreground (accent or semantic color)
- `.ds-badge` — severity/status pill (colored dot + text)
- `.ds-table` — full-width table, fixed layout
- `.ds-pill` — small inline tag
- `.ds-empty` — empty state message

### Typography Scale

| Use | Size |
|---|---|
| Metric value (stat card) | 20px bold |
| Section title | 10px uppercase, letter-spaced |
| Body / list item | 13px |
| Table cell | 12px |
| Subtitle / muted | 11px |

### Responsive Rules

- All grids: `repeat(auto-fit, minmax(Xpx, 1fr))` — never `repeat(N, 1fr)`
- All flex/grid children: `min-width: 0`
- All text: `overflow-wrap: break-word`
- No hardcoded `px` widths on containers

---

## Section 2: Sidebar — Matter Intelligence

### Structure (top to bottom, fully expanded)

1. **Header** — "Matter Intelligence" title, subtle border-bottom separator
2. **Stat Cards** — `auto-fit minmax(90px, 1fr)` grid:
   - Assertions, Open Issues, Gaps, Actors, Quant Facts, LLM Cost, Calls
   - Each card: large `.ds-card-value` + `.ds-card-label`
3. **Coverage Distribution** — `.ds-bar-row` for each issue, accent fill, % label right-aligned
4. **LLM Spend by Tier** — same bar row pattern, amber fill
5. **Weakest Issues** — numbered list, severity dot badge, truncated title with `title` attribute for hover
6. **Gaps & Clarifications** — two sub-sections, each item as a row with colored left-border
7. **Open Work / Actions** — bullet list of pending actions

### Overflow Handling

- Long text: `overflow: hidden; text-overflow: ellipsis; white-space: nowrap` + `title` attribute
- Panel: `overflow-y: auto; max-height: calc(100vh - 120px)` — scrolls independently

### Callback Compatibility

Existing output targets (`overview_md`, `issues_md`, `gaps_md`, `assumptions_md`) are unchanged. All `_fmt_*` function signatures are unchanged — only the returned HTML and the embedded CSS are updated. The `visible=False` components kept for callback wiring are not removed.

---

## Section 3: Per-Tab Data Renderers

### Issues Tab

- Hierarchical tree: each issue as a row with severity badge, title, status pill, action count
- Sub-issues indented 16px with a connecting left border line
- No truncation — text wraps naturally
- Empty state: "No issues found — run an investigation"

### Assertions Tab

- `<table>` with `table-layout: fixed; width: 100%`
- Column widths in `%`: # (5%), Assertion (50%), Source (20%), Confidence (12%), Status (13%)
- `word-break: break-word` on all cells
- Alternating row backgrounds using `var(--background-fill-secondary)`
- Empty state: "No assertions recorded yet"

### Analysis Tab — Sub-panels

**Quant/Financial Facts**
- Card grid `auto-fit minmax(140px, 1fr)`
- Each card: metric name + formatted value + unit
- Color-coded accent: currency = green, percentage = blue, count = neutral

**Timeline**
- Vertical timeline: center dot + left date + right event description
- Stacks to single column on narrow widths
- Date formatted as `DD MMM YYYY`

**Evidence Matrix**
- Grid table: rows = issues, columns = evidence sources
- Cell states: filled dot (covered), empty circle (not covered), half-filled (partial)
- Headers truncated with `title` hover

### Communication Tab

- Actor list with relationship summary cards
- Each relationship: Actor A → relationship type badge → Actor B
- No canvas/SVG — structured relationship rows using `.ds-card` components
- Empty state: "No communication data available"

### Analytics Tab

- LLM usage bar rows: model name + `.ds-bar-row` + cost right-aligned
- Total spend summary `.ds-card` at top
- Call count breakdown as `.ds-pill` tags
- Empty state: "No LLM calls recorded yet"

---

## Section 4: Responsive Layout & Global Fixes

### App Shell

- Main `gr.Row` (tabs + sidebar): both columns get `min_width=0`
- Sidebar column: `scale=1, min_width=260`
- All inner `gr.Column` and `gr.Row`: `min_width=0`

### CSS Global Rules (added to `_css` string)

```css
/* Prevent layout overflow everywhere */
.gradio-container * { box-sizing: border-box; }
.gradio-row, .gradio-column { min-width: 0; }

/* Table safety */
table { width: 100%; table-layout: fixed; }
td, th { overflow-wrap: break-word; word-break: break-word; }

/* Image safety */
img { max-width: 100%; height: auto; }
```

### Preserved Functionality

The following are not touched:
- File upload handlers and `webkitdirectory` JS injection
- All `gr.Button` click callbacks
- All `gr.Dropdown` / `gr.Textbox` inputs
- `_refresh_all()` and correction callbacks
- Server launch configuration (`launch()` args)
- `MatterRepository` and engine integration

---

## Out of Scope

- Adding new features or data sources
- Changing tab names or order
- Modifying backend logic (`engine.py`, `decisions.py`, `repository.py`)
- Canvas/SVG-based graph rendering (not supported in Gradio `gr.HTML`)
- Mobile-first breakpoints (Gradio's own shell is not mobile-optimised)
