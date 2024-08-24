#!/bin/bash

years=(2023)
months=(09 10 11 12 )

base_url="https://api.chess.com/pub/player/doolasux/games"

for year in "${years[@]}"; do
    for month in "${months[@]}"; do
        # Construct URL
        url="${base_url}/${year}/${month}/pgn"
        
        curl $url > ../data/raw_data/games_${month}_${year}.txt
    done
done
