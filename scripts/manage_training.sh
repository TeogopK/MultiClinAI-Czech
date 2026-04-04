#!/bin/bash
# manage_training.sh — monitors pods, downloads completed models, queues next run
# Usage: bash manage_training.sh
# Logs to manage_training.log

SSH_KEY=~/.ssh/id_runpod
REPO=/workspace/ClinicAI
LOCAL_DIR=/c/Users/Teodor.Kostadinov/UNI/ClinicAI/final_models
LOGFILE=/c/Users/Teodor.Kostadinov/UNI/ClinicAI/manage_training.log
STATE_DIR=/c/Users/Teodor.Kostadinov/UNI/ClinicAI/.pod_state

mkdir -p "$LOCAL_DIR" "$STATE_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOGFILE"; }

ssh_cmd() { ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -i "$SSH_KEY" "$@"; }
scp_cmd() { scp -o StrictHostKeyChecking=no -o ConnectTimeout=20 -i "$SSH_KEY" "$@"; }

# ── Queue: "name|entity|rename|log|train_cmd" ─────────────────────────────────

POD1_IP="99.69.17.69"; POD1_PORT="11168"
POD1_QUEUE=(
  "disease-xlmr-os1|disease|best-xlmr-os1-final|disease_xlmr_os1_final.log|PYTHONPATH=src python src/training/train.py --entity_type disease --extra_data_dirs data/augmented/disease --batch_size 64 --epochs 10 --sentence_level --oversample 1 --train_on_all"
  "disease-xlmr-crf|disease|best-xlmr-crf-final|disease_xlmr_crf_final.log|PYTHONPATH=src python src/training/train.py --entity_type disease --use_crf --extra_data_dirs data/augmented/disease --batch_size 32 --epochs 10 --sentence_level --train_on_all"
  "disease-xlmr-para|disease|best-xlmr-para-final|disease_xlmr_para_final.log|PYTHONPATH=src python src/training/train.py --entity_type disease --extra_data_dirs data/augmented/disease data/paraphrase/disease --batch_size 64 --epochs 10 --sentence_level --train_on_all"
  "symptom-xlmr-os1|symptom|best-xlmr-os1-final|symptom_xlmr_os1_final.log|PYTHONPATH=src python src/training/train.py --entity_type symptom --extra_data_dirs data/augmented/symptom --batch_size 64 --epochs 10 --sentence_level --oversample 1 --train_on_all"
)

POD2_IP="213.144.200.243"; POD2_PORT="10946"
POD2_QUEUE=(
  "disease-robeczech-morph|disease|best-robeczech-morph-final|disease_robeczech_final.log|PYTHONPATH=src python src/training/train.py --entity_type disease --model_name ufal/robeczech-base --extra_data_dirs data/augmented/disease --batch_size 64 --epochs 10 --sentence_level --train_on_all"
  "disease-robeczech-para|disease|best-robeczech-para-final|disease_robeczech_para_final.log|PYTHONPATH=src python src/training/train.py --entity_type disease --model_name ufal/robeczech-base --extra_data_dirs data/augmented/disease data/paraphrase/disease --batch_size 64 --epochs 10 --sentence_level --train_on_all"
  "symptom-xlmr-crf|symptom|best-xlmr-crf-final|symptom_xlmr_crf_final.log|PYTHONPATH=src python src/training/train.py --entity_type symptom --use_crf --extra_data_dirs data/augmented/symptom --batch_size 32 --epochs 10 --sentence_level --train_on_all"
  "procedure-xlmr-crf|procedure|best-xlmr-crf-final|procedure_xlmr_crf_final.log|PYTHONPATH=src python src/training/train.py --entity_type procedure --use_crf --extra_data_dirs data/augmented/procedure --batch_size 32 --epochs 10 --sentence_level --train_on_all"
)

POD3_IP="213.144.200.241"; POD3_PORT="13835"
POD3_QUEUE=(
  "symptom-robeczech-os1|symptom|best-robeczech-os1-final|symptom_robeczech_os1_final.log|PYTHONPATH=src python src/training/train.py --entity_type symptom --model_name ufal/robeczech-base --extra_data_dirs data/augmented/symptom --batch_size 64 --epochs 10 --sentence_level --oversample 1 --train_on_all"
  "procedure-robeczech-os1|procedure|best-robeczech-os1-final|procedure_robeczech_os1_final.log|PYTHONPATH=src python src/training/train.py --entity_type procedure --model_name ufal/robeczech-base --extra_data_dirs data/augmented/procedure --batch_size 64 --epochs 10 --sentence_level --oversample 1 --train_on_all"
)

# Initialise state (0 = first run already started externally)
[ -f "$STATE_DIR/pod1_idx" ] || echo 0 > "$STATE_DIR/pod1_idx"
[ -f "$STATE_DIR/pod2_idx" ] || echo 0 > "$STATE_DIR/pod2_idx"
[ -f "$STATE_DIR/pod3_idx" ] || echo 0 > "$STATE_DIR/pod3_idx"

# ── handle_pod pod_id ip port queue_entries... ────────────────────────────────

handle_pod() {
  local pod_id="$1" ip="$2" port="$3"
  shift 3
  local queue=("$@")
  local total=${#queue[@]}
  local state_file="$STATE_DIR/${pod_id}_idx"
  local idx
  idx=$(cat "$state_file")

  if [ "$idx" -ge "$total" ]; then
    log "$pod_id: all $total runs complete ✅"
    return
  fi

  IFS='|' read -r name entity rename logfile cmd <<< "${queue[$idx]}"
  local host="root@$ip -p $port"

  # Check if current run finished
  if ssh_cmd root@"$ip" -p "$port" "grep -q 'Train-on-all complete' $REPO/logs/$logfile 2>/dev/null"; then
    log "$pod_id [$((idx+1))/$total] '$name' DONE ✅"

    # Download model
    log "  Downloading model → final_models/$name/"
    mkdir -p "$LOCAL_DIR/$name"
    scp_cmd -P "$port" -r "root@$ip:$REPO/models/cz-$entity/best/." "$LOCAL_DIR/$name/" \
      && log "  ✅ Model saved to final_models/$name/" \
      || log "  ❌ Model SCP failed"

    # Download log
    scp_cmd -P "$port" "root@$ip:$REPO/logs/$logfile" "$LOCAL_DIR/$name/train.log" 2>/dev/null \
      && log "  ✅ Log saved" || log "  ⚠️  Log SCP failed"

    # Show last loss from log
    local last_loss
    last_loss=$(grep "loss" "$LOCAL_DIR/$name/train.log" 2>/dev/null | grep -v "eval" | tail -1)
    [ -n "$last_loss" ] && log "  Last: $last_loss"

    # Rename best/ on pod so next run doesn't overwrite it
    ssh_cmd root@"$ip" -p "$port" "mv $REPO/models/cz-$entity/best $REPO/models/cz-$entity/$rename 2>/dev/null"
    log "  Renamed on pod: models/cz-$entity/$rename"

    # Advance state and start next run
    local next_idx=$((idx + 1))
    echo "$next_idx" > "$state_file"

    if [ "$next_idx" -lt "$total" ]; then
      IFS='|' read -r next_name next_entity next_rename next_logfile next_cmd <<< "${queue[$next_idx]}"
      log "  Starting next: $next_name"
      ssh_cmd root@"$ip" -p "$port" \
        "mkdir -p $REPO/logs && nohup bash -c 'cd $REPO && $next_cmd > $REPO/logs/$next_logfile 2>&1' & disown"
      sleep 3
      if ssh_cmd root@"$ip" -p "$port" "pgrep -qf 'src/training/train.py'" 2>/dev/null; then
        log "  ✅ $next_name running"
      else
        log "  ⚠️  $next_name may not have started"
      fi
    else
      log "$pod_id: all runs complete ✅"
    fi

  elif ssh_cmd root@"$ip" -p "$port" "pgrep -qf 'src/training/train.py'" 2>/dev/null; then
    local progress
    progress=$(ssh_cmd root@"$ip" -p "$port" "tail -2 $REPO/logs/$logfile 2>/dev/null | grep -oP '\d+(?=%)' | tail -1")
    log "$pod_id [$((idx+1))/$total] '$name' running ${progress:+${progress}%}"
  else
    log "$pod_id [$((idx+1))/$total] '$name' — process not found, checking log..."
    ssh_cmd root@"$ip" -p "$port" "tail -3 $REPO/logs/$logfile 2>/dev/null" 2>/dev/null \
      | while read -r line; do log "  $line"; done
  fi
}

# ── Main loop ─────────────────────────────────────────────────────────────────

log "=== manage_training.sh started — 10 runs across 3 pods ==="

while true; do
  log "────── poll ──────"
  handle_pod "pod1" "$POD1_IP" "$POD1_PORT" "${POD1_QUEUE[@]}"
  handle_pod "pod2" "$POD2_IP" "$POD2_PORT" "${POD2_QUEUE[@]}"
  handle_pod "pod3" "$POD3_IP" "$POD3_PORT" "${POD3_QUEUE[@]}"

  p1=$(cat "$STATE_DIR/pod1_idx")
  p2=$(cat "$STATE_DIR/pod2_idx")
  p3=$(cat "$STATE_DIR/pod3_idx")
  if [ "$p1" -ge "${#POD1_QUEUE[@]}" ] && \
     [ "$p2" -ge "${#POD2_QUEUE[@]}" ] && \
     [ "$p3" -ge "${#POD3_QUEUE[@]}" ]; then
    log "=== ALL 10 RUNS COMPLETE ==="
    break
  fi

  log "Sleeping 5 min..."
  sleep 300
done
