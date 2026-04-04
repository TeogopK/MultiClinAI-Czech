# Run all 10 final models on the test set
# Usage: from repo root: .\run_test_predictions.ps1
$env:PYTHONPATH = "src"

$runs = @(
    @{ entity="disease";   model="disease-xlmr-os1";       crf=$false },
    @{ entity="disease";   model="disease-xlmr-crf";       crf=$true  },
    @{ entity="disease";   model="disease-xlmr-para";      crf=$false },
    @{ entity="disease";   model="disease-robeczech-morph"; crf=$false },
    @{ entity="disease";   model="disease-robeczech-para"; crf=$false },
    @{ entity="symptom";   model="symptom-xlmr-os1";       crf=$false },
    @{ entity="symptom";   model="symptom-xlmr-crf";       crf=$true  },
    @{ entity="symptom";   model="symptom-robeczech-os1";  crf=$false },
    @{ entity="procedure"; model="procedure-xlmr-crf";     crf=$true  },
    @{ entity="procedure"; model="procedure-robeczech-os1"; crf=$false }
)

foreach ($r in $runs) {
    $out = "submission/cz/raw_brat/$($r.model)"
    if (Test-Path "$out/*.ann") {
        $count = (Get-ChildItem "$out/*.ann").Count
        Write-Host "SKIP $($r.model) — $count .ann files already present"
        continue
    }
    Write-Host "=== Running $($r.model) ==="
    $args_list = @(
        "src/inference/predict.py",
        "--entity_type", $r.entity,
        "--model_path", "final_models/$($r.model)",
        "--output_dir", $out,
        "--split", "test",
        "--sentence_level",
        "--device", "cuda"
    )
    if ($r.crf) { $args_list += "--use_crf" }
    python @args_list
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR on $($r.model), exit code $LASTEXITCODE"
        exit 1
    }
    Write-Host "Done: $($r.model)"
}

Write-Host ""
Write-Host "All predictions complete."
