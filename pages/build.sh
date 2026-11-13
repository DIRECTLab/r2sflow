#!/usr/bin/env bash
# Render blog.md to a static site in _site/ (used by .github/workflows/blog-pages.yml).
# Requires pandoc >= 3.0. Run from the repo root: ./pages/build.sh
set -euo pipefail

out="${1:-_site}"
rm -rf "$out"
mkdir -p "$out"

pandoc blog.md \
  --from markdown+tex_math_dollars+implicit_figures \
  --to html5 \
  --standalone \
  --mathjax \
  --toc --toc-depth=2 \
  --section-divs \
  --css style.css \
  --include-in-header pages/header.html \
  --metadata pagetitle="From Noise to Point Clouds" \
  --output "$out/index.html"

cp pages/style.css "$out/"
cp -r figs "$out/"
touch "$out/.nojekyll"
echo "Built $out/index.html"
