# Insurance Paraplanner

A JSON-in/JSON-out advisory workflow over the existing ingested policy and archetype data.

## Run

From the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r insurance_paraplanner\requirements.txt
python -c "import json; from insurance_paraplanner import run_advisory; profile=json.load(open('sample_profile_form.json')); print(json.dumps(run_advisory(profile), indent=2, default=str))"
```

The package also has a command-line entry point. Pass either a JSON file path:

```powershell
python -m insurance_paraplanner example.json
```

or an inline JSON object:

```powershell
python -m insurance_paraplanner '{"Section_A_Know_Your_Client": {}}'
```

Optional database paths can be supplied explicitly:

```powershell
python -m insurance_paraplanner example.json `
	--database ".\data ingestion\policies.db" `
	--chroma-dir ".\data ingestion\chroma_db"
```

The package reads `data ingestion/policies.db` and `data ingestion/chroma_db` by default.
Override those paths by passing `database_path` and `chroma_path` to `run_advisory`.

The workflow derives age from `Date_of_Birth`, uses the midpoint of income ranges, ranks
age-compatible archetypes by marital status and dependant count, calculates deterministic
gaps, and retrieves policy clause evidence. Outputs are advisory only and require human
review.

## Limitations

The current policy ingestion schema does not store policy premiums or detailed feature
columns. The recommender therefore filters by category and age and reports a warning when
premium filtering or feature verification cannot be performed. Do not treat a returned
policy as affordable until premium data is added to the policy ingestion schema.
