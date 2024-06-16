#!/bin/bash

years=(2024)
months=(05 06)

base_url="https://api.chess.com/pub/player/doolasux/games"

for year in "${years[@]}"; do
    for month in "${months[@]}"; do
        # Construct URL
        url="${base_url}/${year}/${month}/pgn"
        
        curl $url > games_${month}_${year}.txt
    done
done
