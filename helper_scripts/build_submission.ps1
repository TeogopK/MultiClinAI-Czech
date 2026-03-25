# Build competition submission TSVs from raw BRAT predictions
# Run after all 10 predictions are complete (run_test_predictions.ps1)
# Usage: .\build_submission.ps1
$env:PYTHONPATH = "src"

$RAW = "submission/cz/raw_brat"
$IND = "submission/cz/individual"
$ENS = "submission/cz/ensemble"

function Convert-ToTSV($pred_dir, $entity, $out_name) {
    $out = "$IND/$out_name"
    python src/brat_to_tsv.py --pred_dir $pred_dir --entity_type $entity --output $out
}

# ── Individual DISEASE ──────────────────────────────────────────────
Convert-ToTSV "$RAW/disease-xlmr-os1"       "disease"   "team_enigma_xlmr-os1_cz_disease.tsv"
Convert-ToTSV "$RAW/disease-xlmr-crf"       "disease"   "team_enigma_xlmr-crf_cz_disease.tsv"
Convert-ToTSV "$RAW/disease-xlmr-para"      "disease"   "team_enigma_xlmr-para_cz_disease.tsv"
Convert-ToTSV "$RAW/disease-robeczech-morph" "disease"  "team_enigma_robeczech-morph_cz_disease.tsv"
Convert-ToTSV "$RAW/disease-robeczech-para" "disease"   "team_enigma_robeczech-para_cz_disease.tsv"

# ── Individual SYMPTOM ──────────────────────────────────────────────
Convert-ToTSV "$RAW/symptom-xlmr-os1"       "symptom"   "team_enigma_xlmr-os1_cz_symptom.tsv"
Convert-ToTSV "$RAW/symptom-xlmr-crf"       "symptom"   "team_enigma_xlmr-crf_cz_symptom.tsv"
Convert-ToTSV "$RAW/symptom-robeczech-os1"  "symptom"   "team_enigma_robeczech-os1_cz_symptom.tsv"

# ── Individual PROCEDURE ────────────────────────────────────────────
Convert-ToTSV "$RAW/procedure-xlmr-crf"     "procedure" "team_enigma_xlmr-crf_cz_procedure.tsv"
Convert-ToTSV "$RAW/procedure-robeczech-os1" "procedure" "team_enigma_robeczech-os1_cz_procedure.tsv"

# ── Ensemble DISEASE ────────────────────────────────────────────────
# Dev F1s: xlmr-os1=0.710, xlmr-crf=0.712, robeczech-morph=0.698, robeczech-para=0.703
Write-Host "=== Ensemble DISEASE ==="
python src/ensemble.py `
    --entity_type disease --split test `
    --pred_dirs "$RAW/disease-xlmr-os1" "$RAW/disease-xlmr-crf" "$RAW/disease-robeczech-morph" "$RAW/disease-robeczech-para" `
    --pred_names "xlmr-os1" "xlmr-crf" "robeczech-morph" "robeczech-para" `
    --model_f1s 0.710 0.712 0.698 0.703 `
    --strategy weighted --threshold 0.4 `
    --output_dir "$RAW/ensemble-disease"
python src/brat_to_tsv.py --pred_dir "$RAW/ensemble-disease" --entity_type disease `
    --output "$ENS/team_enigma_ensemble_cz_disease.tsv"

# ── Ensemble SYMPTOM ────────────────────────────────────────────────
# Dev F1s: xlmr-crf=0.685, robeczech-os1=0.680
Write-Host "=== Ensemble SYMPTOM ==="
python src/ensemble.py `
    --entity_type symptom --split test `
    --pred_dirs "$RAW/symptom-xlmr-crf" "$RAW/symptom-robeczech-os1" `
    --pred_names "xlmr-crf" "robeczech-os1" `
    --model_f1s 0.685 0.680 `
    --strategy weighted --threshold 0.4 `
    --output_dir "$RAW/ensemble-symptom"
python src/brat_to_tsv.py --pred_dir "$RAW/ensemble-symptom" --entity_type symptom `
    --output "$ENS/team_enigma_ensemble_cz_symptom.tsv"

# ── Ensemble PROCEDURE ──────────────────────────────────────────────
# Dev F1s: xlmr-crf=0.739, robeczech-os1=0.735
Write-Host "=== Ensemble PROCEDURE ==="
python src/ensemble.py `
    --entity_type procedure --split test `
    --pred_dirs "$RAW/procedure-xlmr-crf" "$RAW/procedure-robeczech-os1" `
    --pred_names "xlmr-crf" "robeczech-os1" `
    --model_f1s 0.739 0.735 `
    --strategy weighted --threshold 0.4 `
    --output_dir "$RAW/ensemble-procedure"
python src/brat_to_tsv.py --pred_dir "$RAW/ensemble-procedure" --entity_type procedure `
    --output "$ENS/team_enigma_ensemble_cz_procedure.tsv"

Write-Host ""
Write-Host "Submission build complete. Files in submission/cz/individual/ and submission/cz/ensemble/"
Get-ChildItem "$IND/*.tsv" | ForEach-Object { Write-Host "  $($_.Name)" }
Get-ChildItem "$ENS/*.tsv" | ForEach-Object { Write-Host "  ensemble/$($_.Name)" }
