#!/bin/bash

years=(2020 2021 2022 2023 2024)
months=(01 02 03 04 05 06 07 08 09 10 11 12)

base_url="https://api.chess.com/pub/player/doolasux/games"

for year in "${years[@]}"; do
    for month in "${months[@]}"; do
        # Construct URL
        url="${base_url}/${year}/${month}/pgn"
        
        curl $url > games_${month}_${year}.txt
    done
done
