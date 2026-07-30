#!/bin/bash
# Fetch Chess.com PGN exports for doolasux.
# Skips months already downloaded. Skips months with no games (empty response).
#
# Usage:
#   ./fetch_games.sh                      # fetch everything up to current month
#   ./fetch_games.sh 2024 01 2026 03      # fetch a specific range

set -euo pipefail

USERNAME="doolasux"
BASE_URL="https://api.chess.com/pub/player/${USERNAME}/games"
OUTPUT_DIR="$(dirname "$0")/../data/raw_data"

# Parse optional range args
START_YEAR=${1:-2023}
START_MONTH=${2:-9}
END_YEAR=${3:-$(date +%Y)}
END_MONTH=${4:-$(date +%m)}

mkdir -p "$OUTPUT_DIR"

year=$START_YEAR
month=$START_MONTH

while true; do
    month_padded=$(printf "%02d" "$((10#$month))")
    outfile="${OUTPUT_DIR}/games_${month_padded}_${year}.txt"

    if [[ -f "$outfile" && -s "$outfile" ]]; then
        echo "  skip  games_${month_padded}_${year}.txt (already exists)"
    else
        url="${BASE_URL}/${year}/${month_padded}/pgn"
        echo "  fetch $url"
        tmpfile=$(mktemp)
        http_code=$(curl -s -o "$tmpfile" -w "%{http_code}" \
            -H "User-Agent: chess-ai-trainer/1.0" "$url")

        if [[ "$http_code" == "200" ]] && [[ -s "$tmpfile" ]]; then
            mv "$tmpfile" "$outfile"
            game_count=$(grep -c "^\[Event " "$outfile" || true)
            echo "         → saved ${game_count} games"
        else
            rm -f "$tmpfile"
            echo "         → no data (HTTP ${http_code}), skipping"
        fi
    fi

    # Advance month
    if [[ "$month" -eq 12 ]]; then
        month=1
        year=$((year + 1))
    else
        month=$((month + 1))
    fi

    # Stop after end month
    if [[ "$year" -gt "$END_YEAR" ]] || \
       [[ "$year" -eq "$END_YEAR" && "$month" -gt "$((10#$END_MONTH))" ]]; then
        break
    fi
done

echo "Done."
