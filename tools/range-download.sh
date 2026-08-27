#!/usr/bin/env bash
set -euo pipefail

url=$1
output=$2
size=$3
parallel=${4:-24}
chunk_size=${CHUNK_SIZE:-536870912}
parts_dir="${output}.parts"
mkdir -p "$parts_dir" "$(dirname "$output")"

part=0
start=0
while (( start < size )); do
  end=$((start + chunk_size - 1))
  (( end >= size )) && end=$((size - 1))
  part_file=$(printf '%s/%05d.part' "$parts_dir" "$part")
  expected=$((end - start + 1))
  if [[ ! -f "$part_file" ]] || [[ $(stat -c %s "$part_file") -ne $expected ]]; then
    while (( $(jobs -rp | wc -l) >= parallel )); do wait -n; done
    (
      curl -fL --retry 20 --retry-delay 2 --connect-timeout 20 \
        -r "$start-$end" "$url" -o "$part_file.tmp"
      [[ $(stat -c %s "$part_file.tmp") -eq $expected ]]
      mv "$part_file.tmp" "$part_file"
    ) &
  fi
  start=$((end + 1))
  part=$((part + 1))
done
wait

cat "$parts_dir"/*.part >"$output.tmp"
[[ $(stat -c %s "$output.tmp") -eq $size ]]
mv "$output.tmp" "$output"
echo "Downloaded $output ($size bytes)"
