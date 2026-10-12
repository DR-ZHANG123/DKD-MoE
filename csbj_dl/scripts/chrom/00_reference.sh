#!/usr/bin/env bash
# hg38 genome (UCSC) for motif scanning: download once, verify the published md5, decompress, index.
set -euo pipefail
D=/path/to/project016_data/raw/reference/hg38
mkdir -p "$D"; cd "$D"
URL=https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips
[ -s hg38.fa.gz ] || wget -q --tries=10 -O hg38.fa.gz "$URL/hg38.fa.gz"
[ -s md5sum.txt ] || wget -q -O md5sum.txt "$URL/md5sum.txt"
grep " hg38.fa.gz$" md5sum.txt | md5sum -c -
[ -s hg38.fa ] || gunzip -c hg38.fa.gz > hg38.fa
[ -s hg38.fa.fai ] || python -c "import pyfaidx; pyfaidx.Fasta('hg38.fa')"
echo "hg38 ready"
