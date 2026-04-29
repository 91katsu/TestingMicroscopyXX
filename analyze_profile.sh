#!/usr/bin/env bash
# Usage: ./analyze_profile.sh <profile_name.nsys-rep> [title]
# Example: ./analyze_profile.sh profile_result_H200_b16_fp16.nsys-rep "Batch_size = 16, float16"
# Outputs a Markdown table of NVTX ranges + wall-clock, ready to copy into reports.

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <profile.nsys-rep> [title]"
    exit 1
fi

REP="$1"
TITLE="${2:-$(basename "$REP" .nsys-rep)}"
SQL="${REP%.nsys-rep}.sqlite"

if [ ! -f "$REP" ]; then
    echo "File not found: $REP"
    exit 1
fi

# Ensure sqlite is up to date (nsys stats regenerates if needed)
nsys stats --report nvtx_sum --format csv "$REP" > /dev/null 2>&1

# Compute wall-clock from NVTX event timeline
WALL_NS=$(sqlite3 "$SQL" \
    "SELECT MAX(end)-MIN(start) FROM NVTX_EVENTS WHERE text LIKE '%::%';")
WALL_S=$(awk "BEGIN{printf \"%.2f\", $WALL_NS/1e9}")
WALL_MIN=$(awk "BEGIN{m=int($WALL_NS/60e9); s=($WALL_NS-m*60e9)/1e9; printf \"%dm %.0fs\", m, s}")

# Total sum across all ranges (for Time% denominator, matches nsys report)
TOTAL_NS=$(sqlite3 "$SQL" \
    "SELECT SUM(end-start) FROM NVTX_EVENTS WHERE text LIKE '%::%';")

echo "##### $TITLE"
echo
echo "| Stage | Count | Avg (s) | Total (s) | Time(%) |"
echo "|---|---:|---:|---:|---:|"
sqlite3 -separator "|" "$SQL" "
    SELECT
        text,
        COUNT(*),
        printf('%.3f', AVG(end-start)/1e9),
        printf('%.2f', SUM(end-start)/1e9),
        printf('%.1f%%', 100.0 * SUM(end-start) / $TOTAL_NS)
    FROM NVTX_EVENTS
    WHERE text IN ('disk::load_image','cpu::preprocess','mem::ram_to_vram','gpu::inference','mem::vram_to_ram','cpu::assemble','disk::write_output')
    GROUP BY text
    ORDER BY CASE text
        WHEN 'disk::load_image'   THEN 1
        WHEN 'cpu::preprocess'    THEN 2
        WHEN 'mem::ram_to_vram'   THEN 3
        WHEN 'gpu::inference'     THEN 4
        WHEN 'mem::vram_to_ram'   THEN 5
        WHEN 'cpu::assemble'      THEN 6
        WHEN 'disk::write_output' THEN 7
    END;
" | awk -F'|' '{printf "| %s | %s | %s | %s | %s |\n", $1, $2, $3, $4, $5}'

echo
echo "**Wall-clock: ${WALL_S} s (~${WALL_MIN})**"
echo
echo "_Time(%) is relative to the sum of all ranges (${TOTAL_NS}ns ≈ $(awk "BEGIN{printf \"%.0f\", $TOTAL_NS/1e9}") s), not wall-clock. Stages run in parallel, so totals overlap._"

# bash analyze_profile.sh profile_result.nsys-rep
