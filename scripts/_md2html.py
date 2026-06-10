"""Converte report.md in HTML (con MathJax per le formule) per la stampa PDF via Chrome headless."""
import markdown
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
src = (ROOT / 'report.md').read_text(encoding='utf-8')

body = markdown.markdown(src, extensions=['tables', 'fenced_code', 'sane_lists'])

html = f"""<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="utf-8">
<script>
MathJax = {{ tex: {{ inlineMath: [['$', '$']], displayMath: [['$$', '$$']] }} }};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>
  body {{ font-family: 'Segoe UI', Calibri, sans-serif; font-size: 11pt; line-height: 1.45;
         max-width: 18.5cm; margin: 0 auto; color: #1a1a1a; }}
  h1 {{ font-size: 20pt; border-bottom: 2px solid #333; padding-bottom: 4px; }}
  h2 {{ font-size: 15pt; margin-top: 1.4em; border-bottom: 1px solid #bbb; padding-bottom: 2px; }}
  h3 {{ font-size: 12.5pt; margin-top: 1.2em; }}
  table {{ border-collapse: collapse; margin: 0.8em 0; font-size: 9.5pt; }}
  th, td {{ border: 1px solid #999; padding: 3px 8px; text-align: left; }}
  th {{ background: #efefef; }}
  code {{ background: #f3f3f3; padding: 1px 4px; border-radius: 3px; font-size: 9.5pt; }}
  pre {{ background: #f3f3f3; padding: 8px; border-radius: 4px; overflow-x: hidden;
        white-space: pre-wrap; font-size: 9pt; }}
  img {{ max-width: 100%; }}
  blockquote {{ border-left: 3px solid #ccc; margin-left: 0; padding-left: 12px; color: #444; }}
  h1, h2, h3 {{ page-break-after: avoid; }}
  table, img, pre {{ page-break-inside: avoid; }}
</style>
</head>
<body>
{body}
</body>
</html>"""

out = ROOT / 'consegna' / 'relazione.html'
out.write_text(html, encoding='utf-8')
print(f'OK: {out}')
