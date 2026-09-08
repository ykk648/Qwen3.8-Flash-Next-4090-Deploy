#!/usr/bin/env bash
set -euo pipefail

url=$1
output=$2
size=$3
parallel=${4:-24}
chunk_size=${CHUNK_SIZE:-536870912}
parts_dir="${output}.parts"
mkdir -p "$parts_dir" "$(dirname "$output")"

if [[ -f "$output" ]] && [[ $(stat -c %s "$output") -eq $size ]]; then
  echo "Already downloaded $output ($size bytes)"
  exit 0
fi

stop_jobs() {
  local pids
  pids=$(jobs -pr)
  [[ -z "$pids" ]] || kill $pids 2>/dev/null || true
}
trap stop_jobs EXIT INT TERM

download_part() {
  local start=$1
  local end=$2
  local part_file=$3
  local expected=$4
  local tmp_file="${part_file}.tmp"
  local segment_file="${tmp_file}.segment"
  local have remote_start segment_size
  local stalled_attempts=0

  touch "$tmp_file"
  while true; do
    have=$(stat -c %s "$tmp_file")
    if [[ -s "$segment_file" ]]; then
      segment_size=$(stat -c %s "$segment_file")
      if (( have + segment_size > expected )); then
        echo "Oversized interrupted segment: $segment_file" >&2
        return 1
      fi
      cat "$segment_file" >> "$tmp_file"
      rm -f "$segment_file"
      stalled_attempts=0
      continue
    fi
    if (( have == expected )); then
      mv "$tmp_file" "$part_file"
      return 0
    fi
    if (( have > expected )); then
      echo "Oversized temporary part: $tmp_file" >&2
      return 1
    fi

    remote_start=$((start + have))
    curl --fail --location --silent --show-error --http1.1 \
      --connect-timeout 20 --speed-limit 65536 --speed-time 30 \
      --retry 5 --retry-all-errors --retry-delay 2 \
      --range "$remote_start-$end" "$url" --output "$segment_file" || true

    segment_size=0
    [[ ! -f "$segment_file" ]] || segment_size=$(stat -c %s "$segment_file")
    if (( segment_size == 0 )); then
      stalled_attempts=$((stalled_attempts + 1))
      if (( stalled_attempts >= 3 )); then
        echo "No progress after repeated retries for range $remote_start-$end" >&2
        return 1
      fi
      echo "Retrying stalled range $remote_start-$end" >&2
      sleep 2
      continue
    fi
    if (( have + segment_size > expected )); then
      echo "Server returned too many bytes for range $remote_start-$end" >&2
      return 1
    fi
    cat "$segment_file" >> "$tmp_file"
    rm -f "$segment_file"
    stalled_attempts=0
  done
}

part=0
start=0
while (( start < size )); do
  end=$((start + chunk_size - 1))
  (( end >= size )) && end=$((size - 1))
  part_file=$(printf '%s/%05d.part' "$parts_dir" "$part")
  expected=$((end - start + 1))
  if [[ ! -f "$part_file" ]] || [[ $(stat -c %s "$part_file") -ne $expected ]]; then
    while (( $(jobs -rp | wc -l) >= parallel )); do wait -n; done
    download_part "$start" "$end" "$part_file" "$expected" &
  fi
  start=$((end + 1))
  part=$((part + 1))
done
wait

cat "$parts_dir"/*.part >"$output.tmp"
[[ $(stat -c %s "$output.tmp") -eq $size ]]
mv "$output.tmp" "$output"
echo "Downloaded $output ($size bytes)"
