#!/bin/bash
# Download MovieLens-1M into data/ml-1m/
set -e
mkdir -p data
cd data
curl -O https://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip -o ml-1m.zip
rm ml-1m.zip
echo "Ready: data/ml-1m/ratings.dat"
