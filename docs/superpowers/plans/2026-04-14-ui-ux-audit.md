# UI/UX Audit & Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the entire Gradio dashboard fully theme-adaptive (light/dark), responsive (no content cut off), and every data type properly rendered, without removing any functionality.

**Architecture:** One `_DS_CSS` constant (embedded `<style>` block, CSS variables throughout) replaces the old `_SIDEBAR_STYLE`. The `_css` string is overhauled to replace all hardcoded hex colors with Gradio CSS variables. The assertions accordion is upgraded from a Markdown table to a proper HTML table.

**Tech Stack:** Python 3, Gradio 6.12.0, HTML/CSS (no JS changes), `src/irys/ui/app.py` only.

---

## File Map

| File | What changes |
|---|---|
| `src/irys/ui/app.py:456-525` | Replace `_SIDEBAR_STYLE` with `_DS_CSS` |
| `src/irys/ui/app.py:528-692` | `_fmt_overview_panel` — swap constant name |
| `src/irys/ui/app.py:1554-1581` | `_fmt_assertions` — Markdown table → HTML table |
| `src/irys/ui/app.py:2526-2844` | `_css` — CSS vars, remove dark navy block, add responsive rules |
| `src/irys/ui/app.py:3043` | `assertions_md` component — `gr.Markdown` → `gr.HTML` |

No other files change.

---

## Task 1: Replace `_SIDEBAR_STYLE` with `_DS_CSS`

**Files:**
- Modify: `src/irys/ui/app.py:456-525`

This replaces the embedded `<style>` block with a theme-adaptive version that uses Gradio CSS variables everywhere. Key fixes: (1) stat card grid changes from `repeat(3,1fr)` (overflows in narrow columns) to `repeat(auto-fit,minmax(90px,1fr))`; (2) all hex colors become CSS variable calls.

- [ ] **Step 1: Remove `_SIDEBAR_STYLE` and write `_DS_CSS`**

Replace lines 456–525 (the entire `_SIDEBAR_STYLE = """<style>..."""` block) with:

```python
_DS_CSS = """<style>
.intel-panel{
  background:var(--background-fill-secondary,#f8fafc);
  border:1px solid var(--block-border-color,#e2e8f0);
  border-radius:14px;padding:16px;
  box-shadow:0 2px 12px rgba(0,0,0,.06);
  display:flex;flex-direction:column;gap:10px;
  width:100%;box-sizing:border-box;overflow:hidden;
}
.intel-panel-title{
  font-size:11px;font-weight:700;
  color:var(--body-text-color,#0f172a);
  letter-spacing:.08em;text-transform:uppercase;
  padding-bottom:6px;
  border-bottom:1px solid var(--block-border-color,#e2e8f0);
  margin-bottom:2px;
}
.intel-panel .viz-shell{display:flex;flex-direction:column;gap:8px;}
.intel-panel .viz-card-grid{
  display:grid;
  grid-template-columns:repeat(auto-fit,minmax(90px,1fr));
  gap:6px;
}
.intel-panel .viz-card{
  border:1px solid var(--block-border-color,#e2e8f0);
  border-radius:10px;padding:8px 6px;
  background:var(--background-fill-primary,#fff);
  overflow:hidden;
}
.intel-panel .viz-card.tone-amber{border-color:rgba(217,119,6,.4);}
.intel-panel .viz-card.tone-green{border-color:rgba(21,128,61,.4);}
.intel-panel .viz-card.tone-red{border-color:rgba(185,28,28,.4);}
.intel-panel .viz-card-title{
  font-size:9px;letter-spacing:.07em;text-transform:uppercase;
  color:var(--body-text-color-subdued,#64748b);margin-bottom:2px;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
}
.intel-panel .viz-card-value{
  font-size:18px;font-weight:700;
  color:var(--body-text-color,#0f172a);line-height:1.1;
}
.intel-panel .viz-card-detail{
  font-size:10px;color:var(--body-text-color-subdued,#64748b);margin-top:3px;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
}
.intel-panel .viz-two-col{display:grid;grid-template-columns:1fr;gap:6px;}
.intel-panel .viz-panel{
  border:1px solid var(--block-border-color,#e2e8f0);
  border-radius:10px;padding:10px;
  background:var(--background-fill-primary,#fff);
}
.intel-panel .viz-panel-title{
  font-size:11px;font-weight:700;
  color:var(--body-text-color,#0f172a);margin-bottom:6px;
}
.intel-panel .viz-subtitle{
  font-size:10px;font-weight:700;
  color:var(--body-text-color-subdued,#64748b);margin-bottom:3px;
}
.intel-panel .viz-footnote{font-size:10px;color:var(--body-text-color-subdued,#64748b);margin-top:4px;}
.intel-panel .viz-bar-row{
  display:flex;flex-direction:column;gap:2px;
  padding:3px 0;
  border-bottom:1px solid var(--block-border-color,#e2e8f0);
}
.intel-panel .viz-bar-label{
  font-size:10px;color:var(--body-text-color,#334155);
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
}
.intel-panel .viz-bar-track{
  height:5px;border-radius:999px;
  background:var(--block-border-color,#e2e8f0);
  overflow:hidden;margin:2px 0;
}
.intel-panel .viz-bar-fill{height:100%;border-radius:999px;}
.intel-panel .viz-bar-fill.tone-green{background:#16a34a;}
.intel-panel .viz-bar-fill.tone-amber{background:#d97706;}
.intel-panel .viz-bar-fill.tone-blue{background:var(--color-accent,#2563eb);}
.intel-panel .viz-bar-meta{font-size:9px;color:var(--body-text-color-subdued,#64748b);}
.intel-panel .viz-list-row{
  display:flex;justify-content:space-between;align-items:flex-start;
  gap:6px;padding:5px 0;
  border-bottom:1px solid var(--block-border-color,#e2e8f0);
  font-size:11px;color:var(--body-text-color,#334155);
}
.intel-panel .viz-list-row strong{
  white-space:nowrap;color:var(--body-text-color-subdued,#64748b);
}
.intel-panel .viz-list-columns{display:grid;grid-template-columns:1fr 1fr;gap:6px;}
.intel-panel .viz-list-columns ul{
  margin:3px 0 0;padding-left:12px;
  font-size:10px;color:var(--body-text-color-subdued,#64748b);
}
.intel-panel .viz-list-columns li{margin-bottom:3px;}
.intel-panel .viz-empty{
  border:1px dashed var(--block-border-color,#cbd5e1);
  border-radius:8px;padding:10px;
  color:var(--body-text-color-subdued,#64748b);font-size:11px;
}
</style>"""
```

- [ ] **Step 2: Verify the constant is syntactically valid**

```bash
cd /home/ubuntu/legal-rlm && python3 -c "from irys.ui.app import _DS_CSS; print('OK', len(_DS_CSS))"
```

Expected: `OK` followed by a character count > 1000.

- [ ] **Step 3: Commit**

```bash
cd /home/ubuntu/legal-rlm
git add src/irys/ui/app.py
git commit -m "$(cat <<'EOF'
Replace _SIDEBAR_STYLE with _DS_CSS: CSS variables, fix card grid overflow

Committed by Devansh
EOF
)"
```

---

## Task 2: Update `_fmt_overview_panel` to use `_DS_CSS`

**Files:**
- Modify: `src/irys/ui/app.py:528-692`

Two lines in this function reference `_SIDEBAR_STYLE`. Replace both with `_DS_CSS`.

- [ ] **Step 1: Replace references**

At line ~531 (inside the early-return branch):
```python
# OLD:
        return (
            _SIDEBAR_STYLE
            + "<div class='intel-panel'>"
# NEW:
        return (
            _DS_CSS
            + "<div class='intel-panel'>"
```

At line ~633 (inside the main return):
```python
# OLD:
    return (
        _SIDEBAR_STYLE
        + "<div class='intel-panel'>"
# NEW:
    return (
        _DS_CSS
        + "<div class='intel-panel'>"
```

- [ ] **Step 2: Verify import is clean**

```bash
cd /home/ubuntu/legal-rlm && python3 -c "from irys.ui.app import _fmt_overview_panel; print(_fmt_overview_panel({})[:80])"
```

Expected: output starting with `<style>` and containing `intel-panel`.

- [ ] **Step 3: Commit**

```bash
cd /home/ubuntu/legal-rlm
git add src/irys/ui/app.py
git commit -m "$(cat <<'EOF'
_fmt_overview_panel: use _DS_CSS instead of removed _SIDEBAR_STYLE

Committed by Devansh
EOF
)"
```

---

## Task 3: Overhaul `_css` — CSS variables, remove dark navy intel-panel block

**Files:**
- Modify: `src/irys/ui/app.py:2526-2844`

Replace the entire `_css = """..."""` block (lines 2526–2844) with the version below. Key changes:
- All `viz-*` class hex colors → CSS variables (`var(--body-text-color)`, `var(--background-fill-primary)`, etc.)
- Remove the `.intel-panel { background: #0f172a; ... }` dark navy block and all its overrides (lines 2726–2843 in the current file) — `_DS_CSS` now owns these, theme-adaptively
- `.intelligence-sidebar` simplified to column-transparency only
- Global responsive rules added at the bottom

- [ ] **Step 1: Replace the entire `_css` block**

Find the line `_css = """` (currently at line 2526) and replace the entire block up to and including its closing `"""` with:

```python
_css = """
    .mono textarea { font-family: 'JetBrains Mono', monospace; font-size: 12px; }
    .status-bar textarea { font-weight: 600; font-size: 13px; }
    .compact-id { font-size: 11px !important; }
    .compact-id textarea { font-size: 11px; color: var(--body-text-color-subdued, #888); }
    footer { display: none !important; }

    /* ── Matter workspace ─────────────────────────────────── */
    .matter-card {
        border: 1px solid var(--block-border-color, #dbe4ef);
        border-radius: 14px; padding: 16px 18px;
        background: var(--background-fill-primary, #fff);
        box-shadow: 0 2px 10px rgba(15,23,42,0.06); margin-bottom: 2px;
    }
    .matter-card-header {
        display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;
    }
    .matter-card-name { font-size: 15px; font-weight: 700; color: var(--body-text-color, #0f172a); }
    .matter-card-badge {
        font-size: 12px; font-weight: 600; color: var(--color-accent, #2563eb);
        background: rgba(37,99,235,0.08); border-radius: 999px; padding: 2px 10px;
    }
    .matter-card-files { display: flex; flex-direction: column; gap: 4px; }
    .matter-file-chip {
        display: flex; align-items: center; gap: 8px; font-size: 12px;
        color: var(--body-text-color, #334155);
        background: var(--background-fill-secondary, #f8fafc);
        border: 1px solid var(--block-border-color, #e2e8f0);
        border-radius: 8px; padding: 5px 10px;
    }
    .matter-file-icon { color: var(--body-text-color-subdued, #64748b); flex-shrink: 0; }
    .matter-empty-state {
        border: 2px dashed var(--block-border-color, #cbd5e1);
        border-radius: 14px; padding: 32px 20px;
        text-align: center; background: var(--background-fill-secondary, #f8fafc);
    }
    .matter-empty-title { font-size: 15px; font-weight: 600; color: var(--body-text-color-subdued, #64748b); margin-bottom: 6px; }
    .matter-empty-sub { font-size: 13px; color: var(--body-text-color-subdued, #94a3b8); }
    .ws-status { font-size: 12px; border-radius: 8px; padding: 7px 12px; margin-top: 4px; }
    .ws-ok  { color: #15803d; background: #f0fdf4; border: 1px solid #bbf7d0; }
    .ws-err { color: #b91c1c; background: #fef2f2; border: 1px solid #fecaca; }
    .ws-info{ color: #1d4ed8; background: #eff6ff; border: 1px solid #bfdbfe; }
    .gap-highlight { background: #fef3c7; border-radius: 6px; padding: 8px; }

    /* ── Shared viz shell ─────────────────────────────────── */
    .viz-shell { display: flex; flex-direction: column; gap: 12px; min-width: 0; }
    .viz-empty {
        border: 1px dashed var(--block-border-color, #cbd5e1); border-radius: 12px; padding: 14px;
        color: var(--body-text-color-subdued, #64748b);
        background: var(--background-fill-secondary, #f8fafc);
    }

    /* ── Stat cards ───────────────────────────────────────── */
    .viz-card-grid {
        display: grid; gap: 10px;
        grid-template-columns: repeat(auto-fit, minmax(135px, 1fr));
    }
    .viz-card {
        border: 1px solid var(--block-border-color, #dbe4ef);
        border-radius: 14px; padding: 12px 14px;
        background: var(--background-fill-primary, #fff);
        box-shadow: 0 2px 8px rgba(15,23,42,0.05);
        min-width: 0;
    }
    .viz-card.tone-amber { border-color: rgba(217,119,6,0.25); }
    .viz-card.tone-green { border-color: rgba(21,128,61,0.25); }
    .viz-card.tone-red   { border-color: rgba(185,28,28,0.25); }
    .viz-card-title {
        font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase;
        color: var(--body-text-color-subdued, #64748b);
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    .viz-card-value {
        font-size: 24px; font-weight: 700;
        color: var(--body-text-color, #0f172a); margin-top: 4px;
    }
    .viz-card-detail {
        font-size: 12px; color: var(--body-text-color-subdued, #475569); margin-top: 6px;
    }

    /* ── Two-column grid ─────────────────────────────────── */
    .viz-two-col {
        display: grid; gap: 12px;
        grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
        min-width: 0;
    }

    /* ── Panel card ──────────────────────────────────────── */
    .viz-panel {
        border: 1px solid var(--block-border-color, #dbe4ef);
        border-radius: 14px; padding: 14px;
        background: var(--background-fill-primary, #fff);
        min-width: 0;
    }
    .viz-panel-title {
        font-size: 13px; font-weight: 700;
        color: var(--body-text-color, #0f172a); margin-bottom: 10px;
    }
    .viz-subtitle {
        font-size: 12px; font-weight: 700;
        color: var(--body-text-color-subdued, #475569); margin-bottom: 6px;
    }
    .viz-footnote { font-size: 11px; color: var(--body-text-color-subdued, #64748b); margin-top: 8px; }

    /* ── List rows ───────────────────────────────────────── */
    .viz-list-row {
        display: flex; justify-content: space-between; align-items: flex-start;
        gap: 12px; padding: 8px 0;
        border-bottom: 1px solid var(--block-border-color, #eef2f7);
        font-size: 12px; color: var(--body-text-color, #334155);
        min-width: 0;
    }
    .viz-list-row span, .viz-list-row strong { white-space: normal; word-break: break-word; min-width: 0; }
    .viz-list-row:last-child { border-bottom: none; }
    .viz-list-columns {
        display: grid; gap: 14px;
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    }
    .viz-list-columns ul { margin: 0; padding-left: 18px; color: var(--body-text-color, #334155); }
    .viz-list-columns li { margin-bottom: 6px; }

    /* ── Bar rows ────────────────────────────────────────── */
    .viz-bar-row {
        display: grid; gap: 8px; align-items: center;
        grid-template-columns: minmax(80px, 1fr) minmax(80px, 2fr) minmax(80px, auto);
        margin-bottom: 8px; min-width: 0;
    }
    .viz-bar-label {
        font-size: 12px; color: var(--body-text-color, #334155);
        white-space: normal; word-break: break-word; min-width: 0;
    }
    .viz-bar-track {
        height: 10px; border-radius: 999px;
        background: var(--block-border-color, #e2e8f0); overflow: hidden; min-width: 0;
    }
    .viz-bar-fill { height: 100%; border-radius: 999px; }
    .viz-bar-fill.tone-blue  { background: linear-gradient(90deg, var(--color-accent,#2563eb), #38bdf8); }
    .viz-bar-fill.tone-amber { background: linear-gradient(90deg, #d97706, #f59e0b); }
    .viz-bar-fill.tone-green { background: linear-gradient(90deg, #15803d, #22c55e); }
    .viz-bar-fill.tone-red   { background: linear-gradient(90deg, #b91c1c, #ef4444); }
    .viz-bar-meta { font-size: 12px; color: var(--body-text-color-subdued, #64748b); text-align: right; min-width: 0; word-break: break-word; }

    /* ── Issues tree ─────────────────────────────────────── */
    .issues-stack { display: flex; flex-direction: column; gap: 10px; min-width: 0; }
    .issue-row {
        padding: 10px 12px 12px calc(12px + var(--issue-indent));
        border: 1px solid var(--block-border-color, #e2e8f0); border-radius: 12px;
        background: var(--background-fill-primary, #fff); min-width: 0;
    }
    .issue-head { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; min-width: 0; }
    .proof-pill {
        border-radius: 999px; padding: 2px 8px; font-size: 10px;
        font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; flex-shrink: 0;
    }
    .proof-strong { background: rgba(21,128,61,0.12); color: #166534; }
    .proof-partial { background: rgba(217,119,6,0.12); color: #b45309; }
    .proof-weak    { background: rgba(249,115,22,0.12); color: #c2410c; }
    .proof-gap     { background: rgba(185,28,28,0.12); color: #b91c1c; }
    .proof-none    { background: rgba(148,163,184,0.18); color: var(--body-text-color-subdued,#475569); }
    .issue-title {
        flex: 1; min-width: 0; font-size: 13px; font-weight: 600;
        color: var(--body-text-color, #0f172a); white-space: normal; word-break: break-word;
    }
    .issue-pct { font-size: 12px; color: var(--body-text-color-subdued, #475569); flex-shrink: 0; }
    .issue-track {
        height: 8px; border-radius: 999px;
        background: var(--block-border-color, #e2e8f0); overflow: hidden;
    }
    .issue-fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg, var(--color-accent,#1d4ed8), #22c55e); }
    .issue-meta { font-size: 12px; color: var(--body-text-color-subdued, #64748b); margin-top: 8px; }

    /* ── Timeline ────────────────────────────────────────── */
    .timeline-list { position: relative; display: flex; flex-direction: column; gap: 12px; min-width: 0; }
    .timeline-item {
        display: grid; gap: 12px; align-items: start;
        grid-template-columns: 100px 18px minmax(0, 1fr);
    }
    .timeline-date {
        font-size: 12px; font-weight: 700;
        color: var(--body-text-color, #334155); padding-top: 2px; word-break: break-word;
    }
    .timeline-line { position: relative; min-height: 56px; }
    .timeline-line::before {
        content: ''; position: absolute; left: 8px; top: 0; bottom: -12px;
        width: 2px; background: var(--block-border-color, #dbe4ef);
    }
    .timeline-dot {
        position: absolute; left: 2px; top: 6px; width: 14px; height: 14px;
        border-radius: 50%; background: var(--color-accent, #2563eb);
        box-shadow: 0 0 0 4px rgba(37,99,235,0.12);
    }
    .timeline-body {
        border: 1px solid var(--block-border-color, #dbe4ef); border-radius: 12px; padding: 10px 12px;
        background: var(--background-fill-primary, #fff); min-width: 0;
    }
    .timeline-title {
        font-size: 13px; font-weight: 600; color: var(--body-text-color, #0f172a);
        white-space: normal; word-break: break-word;
    }
    .timeline-meta {
        font-size: 12px; color: var(--body-text-color-subdued, #64748b);
        margin-top: 6px; white-space: normal; word-break: break-word;
    }

    /* ── Evidence / analytics tables ─────────────────────── */
    .matrix-wrap { overflow: auto; max-width: 100%; }
    .matrix-wrap-heatmap {
        overflow: auto; max-width: 100%; max-height: 72vh;
        border: 1px solid var(--block-border-color, #dbe4ef); border-radius: 12px;
        background: var(--background-fill-primary, #fff);
    }
    .matrix-table, .analytics-table {
        width: 100%; border-collapse: separate; border-spacing: 0; font-size: 12px;
    }
    .matrix-table th, .matrix-table td, .analytics-table th, .analytics-table td {
        border-bottom: 1px solid var(--block-border-color, #e2e8f0);
        padding: 8px 10px; text-align: left;
        white-space: normal; word-break: break-word; vertical-align: top;
    }
    .matrix-table thead th, .analytics-table thead th {
        position: sticky; top: 0; z-index: 1;
        background: var(--background-fill-secondary, #f8fafc);
        color: var(--body-text-color, #334155);
        font-weight: 700;
    }
    .matrix-cell { min-width: 52px; text-align: center !important; font-weight: 700; color: var(--body-text-color, #0f172a); }
    .evidence-matrix-table { width: max-content; min-width: max-content; table-layout: fixed; }
    .evidence-matrix-table thead th { min-width: 170px; max-width: 220px; z-index: 3; }
    .evidence-matrix-table thead th:first-child {
        min-width: 220px; max-width: 300px; left: 0; z-index: 5;
        box-shadow: 2px 0 0 var(--block-border-color, #dbe4ef);
    }
    .evidence-matrix-table tbody th {
        position: sticky; left: 0; min-width: 220px; max-width: 300px; z-index: 2;
        background: var(--background-fill-secondary, #f8fafc);
        box-shadow: 2px 0 0 var(--block-border-color, #dbe4ef);
    }
    .evidence-matrix-table td.matrix-cell { min-width: 72px; width: 72px; text-align: center !important; }

    /* ── Communication graph ─────────────────────────────── */
    .comm-graph {
        width: 100%; height: auto;
        border: 1px solid var(--block-border-color, #dbe4ef); border-radius: 14px;
        background: var(--background-fill-secondary, #f8fafc); overflow: visible;
    }
    .comm-actor-node { fill: var(--color-accent, #1d4ed8); opacity: 0.9; }
    .comm-doc-node { fill: #0f766e; opacity: 0.85; }
    .comm-label { font-size: 11px; fill: var(--body-text-color, #334155); font-family: 'Inter', sans-serif; }
    .comm-label-left { text-anchor: start; }

    /* ── Expandable detail ───────────────────────────────── */
    .viz-detail { border-top: 1px solid var(--block-border-color, #e2e8f0); padding: 10px 0; }
    .viz-detail:first-child { border-top: none; }
    .viz-detail summary { cursor: pointer; font-weight: 600; color: var(--body-text-color, #0f172a); }
    .viz-detail-block { margin-top: 10px; font-size: 12px; color: var(--body-text-color, #334155); }
    .viz-detail-block ul { margin: 6px 0 0 0; padding-left: 18px; }
    .viz-detail-block li { margin-bottom: 6px; }

    /* ── Sidebar column transparency ─────────────────────── */
    .intelligence-sidebar {
        --block-background-fill: transparent;
        --block-border-color: transparent;
        --block-border-width: 0px;
        --block-shadow: none;
        --block-padding: 0px;
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
    }
    .intelligence-sidebar > div,
    .intelligence-sidebar > div > div {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
    }

    /* ── Global responsive safety net ───────────────────── */
    .gradio-container * { box-sizing: border-box; }
    .gradio-container .gr-row, .gradio-container .gr-column { min-width: 0; }
    table { width: 100%; }
    td, th { overflow-wrap: break-word; word-break: break-word; }
    img { max-width: 100%; height: auto; }
    """
```

- [ ] **Step 2: Verify the module loads**

```bash
cd /home/ubuntu/legal-rlm && python3 -c "from irys.ui.app import _css; print('OK', len(_css))"
```

Expected: `OK` followed by character count > 3000.

- [ ] **Step 3: Commit**

```bash
cd /home/ubuntu/legal-rlm
git add src/irys/ui/app.py
git commit -m "$(cat <<'EOF'
Overhaul _css: CSS variables throughout, remove dark navy intel-panel block, add responsive rules

Committed by Devansh
EOF
)"
```

---

## Task 4: Convert `_fmt_assertions` from Markdown table to HTML and update component

**Files:**
- Modify: `src/irys/ui/app.py:1554-1581` (`_fmt_assertions` function)
- Modify: `src/irys/ui/app.py:3043` (`assertions_md` component declaration)

The Markdown table is hard to read and can't be styled. Replace with an HTML table using the existing `.analytics-table` class (now defined with CSS variables in `_css`).

- [ ] **Step 1: Rewrite `_fmt_assertions`**

Replace lines 1554–1581 with:

```python
def _fmt_assertions(assertions: list) -> str:
    if not assertions:
        return "<div class='viz-empty'>No assertions recorded yet.</div>"
    rows_html = ""
    for a in assertions:
        assertion_id = _escape(a.get("id", "?"))
        prop = _escape(a.get("proposition_text") or "")
        state = _escape(a.get("belief_state") or "—")
        conf = f"{float(a.get('confidence', 0)):.2f}" if a.get("confidence") is not None else "—"
        src_roles = a.get("source_roles", [])
        if len(src_roles) > 1:
            best = min(
                src_roles,
                key=lambda r: list(_TRUST_ICONS).index(r.upper())
                if r.upper() in _TRUST_ICONS else 99,
            )
            icon = _trust_icon(best)
            src = _escape(f"MULTI[{','.join(src_roles)}]")
        elif src_roles:
            icon = _trust_icon(src_roles[0])
            src = _escape(src_roles[0])
        else:
            src_role = a.get("source_role") or a.get("primary_source_role") or "—"
            icon = _trust_icon(src_role)
            src = _escape(src_role)
        speech = _escape(a.get("speech_act") or a.get("primary_speech_act") or "—")
        rows_html += (
            "<tr>"
            f"<td style='text-align:center'>{icon}</td>"
            f"<td>{prop}</td>"
            f"<td>{state}</td>"
            f"<td style='text-align:right'>{conf}</td>"
            f"<td>{src}</td>"
            f"<td>{speech}</td>"
            f"<td><code style='font-size:10px'>{assertion_id}</code></td>"
            "</tr>"
        )
    return (
        "<div class='matrix-wrap'>"
        "<table class='analytics-table' style='table-layout:fixed;width:100%'>"
        "<colgroup>"
        "<col style='width:36px'>"
        "<col style='width:40%'>"
        "<col style='width:10%'>"
        "<col style='width:8%'>"
        "<col style='width:12%'>"
        "<col style='width:12%'>"
        "<col style='width:18%'>"
        "</colgroup>"
        "<thead><tr>"
        "<th></th><th>Proposition</th><th>State</th>"
        "<th>Conf</th><th>Source</th><th>Speech</th><th>ID</th>"
        "</tr></thead>"
        "<tbody>" + rows_html + "</tbody>"
        "</table></div>"
    )
```

- [ ] **Step 2: Change `assertions_md` from `gr.Markdown` to `gr.HTML`**

Find (around line 3043):
```python
            assertions_md = gr.Markdown("*Facts will appear here after an investigation.*")
```

Replace with:
```python
            assertions_md = gr.HTML("<div class='viz-empty'>Facts will appear here after an investigation.</div>")
```

- [ ] **Step 3: Verify the function works**

```bash
cd /home/ubuntu/legal-rlm && python3 -c "
from irys.ui.app import _fmt_assertions
print(_fmt_assertions([])[:60])
sample = [{'id':'a1','proposition_text':'Plaintiff alleges breach','belief_state':'alleged','confidence':0.9,'source_roles':['plaintiff_statement'],'speech_act':'allegation'}]
out = _fmt_assertions(sample)
assert 'analytics-table' in out
assert 'Plaintiff alleges breach' in out
print('HTML assertions OK')
"
```

Expected: output containing `viz-empty` then `HTML assertions OK`.

- [ ] **Step 4: Commit**

```bash
cd /home/ubuntu/legal-rlm
git add src/irys/ui/app.py
git commit -m "$(cat <<'EOF'
Convert assertions display from Markdown table to responsive HTML table

Committed by Devansh
EOF
)"
```

---

## Task 5: Smoke-test the app end-to-end

**Goal:** Confirm the app starts, the sidebar renders with theme-adaptive styles, and the accordions open without layout breaks. This is a manual visual check — no automated test needed for CSS.

- [ ] **Step 1: Kill any running server and restart**

```bash
pkill -f "irys.ui.app" 2>/dev/null; sleep 1
cd /home/ubuntu/legal-rlm && python3 -m irys.ui.app > /tmp/ui_server.log 2>&1 &
sleep 5
cat /tmp/ui_server.log
```

Expected: log shows `Running on local URL:  http://0.0.0.0:7860` (or similar), no import errors.

- [ ] **Step 2: Check for Python errors in the log**

```bash
grep -i "error\|traceback\|exception" /tmp/ui_server.log | head -20
```

Expected: no output (zero errors).

- [ ] **Step 3: Confirm sidebar HTML contains CSS variables (not hardcoded dark hex)**

```bash
cd /home/ubuntu/legal-rlm && python3 -c "
from irys.ui.app import _fmt_overview_panel
html = _fmt_overview_panel({})
assert '--background-fill-secondary' in html, 'CSS var missing'
assert '#0f172a' not in html, 'Dark navy hardcode found in sidebar HTML'
print('Sidebar CSS OK')
"
```

Expected: `Sidebar CSS OK`.

- [ ] **Step 4: Confirm assertions HTML output is correct type**

```bash
cd /home/ubuntu/legal-rlm && python3 -c "
from irys.ui.app import _fmt_assertions
out = _fmt_assertions([])
assert '<div' in out, 'Not HTML'
assert 'viz-empty' in out
print('Assertions HTML OK')
"
```

Expected: `Assertions HTML OK`.

- [ ] **Step 5: (Manual) Open http://localhost:7860 in a browser**

Check:
- [ ] Sidebar shows "Matter Intelligence" panel with theme-matching background (not dark navy in light mode)
- [ ] Sidebar stat card grid does not overflow (cards wrap or shrink, not cut off)
- [ ] Open the "Extracted Facts" accordion — shows empty state message, not a Markdown table
- [ ] Toggle dark/light theme in Gradio settings — sidebar background and text update correctly
- [ ] Open "Timeline", "Evidence Matrix", "Communication Graph", "LLM Analytics" accordions — all show empty-state messages in correct theme colors (muted text, not hardcoded dark hex)

- [ ] **Step 6: Final commit if manual check passes**

```bash
cd /home/ubuntu/legal-rlm
git add src/irys/ui/app.py
git commit -m "$(cat <<'EOF'
UI/UX audit complete: theme-adaptive design system, responsive layout, HTML assertions

Committed by Devansh
EOF
)" 2>/dev/null || echo "Nothing to commit - all changes already committed"
```

---

## Self-Review

### Spec coverage check

| Spec requirement | Task that covers it |
|---|---|
| One `_DS_CSS` constant, all colors CSS variables | Task 1 |
| Sidebar: `auto-fit minmax(90px, 1fr)` stat card grid | Task 1 |
| `_css` strip hardcoded colors | Task 3 |
| Remove dark navy `.intel-panel` block | Task 3 |
| `.intelligence-sidebar` simplified to column transparency | Task 3 |
| Global `min-width: 0`, `overflow-wrap: break-word`, `box-sizing: border-box` | Task 3 |
| Assertions: HTML table, `table-layout: fixed`, `%` column widths | Task 4 |
| `assertions_md` component → `gr.HTML` | Task 4 |
| All function signatures unchanged | All tasks ✓ |
| All callbacks unchanged | All tasks ✓ |
| Zero functionality removed | All tasks ✓ |

### Placeholder scan

No TBDs, TODOs, or "similar to Task N" references found.

### Type consistency

- `_DS_CSS` is a `str` — same as `_SIDEBAR_STYLE` was, same usage pattern
- `_fmt_assertions` still returns `str` (HTML now, not Markdown) — component type updated to match
- All other `_fmt_*` signatures unchanged

### Gaps

Per spec, `_fmt_issues_panel`, `_fmt_timeline_panel`, `_fmt_evidence_matrix_panel`, `_fmt_communication_map_panel`, `_fmt_llm_analytics_panel`, `_fmt_quant_panel` already use only `viz-*` classes which are now CSS-variable-based in `_css`. Since these panels are in accordions (collapsed by default), `_css` is loaded before the user can open them — no timing issue, no embedded style block needed.
