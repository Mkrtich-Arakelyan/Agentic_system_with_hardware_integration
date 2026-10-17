# Contextual print-tuning validation

`app.print_tuning.validation` validates already parsed responses using loaded
data. It performs no filesystem access, retries, model calls, printing, or session
updates. `TuningValidationError.errors` contains individual actionable messages.

```python
from app.print_tuning import ValidationContext, validate_response

# response: GammaProposalResponse, config: loaded tuning_config.json
context = ValidationContext(
    expected_stage="propose_gamma",
    selected_experiment_ids=frozen_ids,
    config=config,
    reference_mappings=current_complete_mappings,
)
validated_response = validate_response(response, context)
```

For a usable review, winner resolution precedes validation of the next proposals:

```python
from app.print_tuning import resolve_review_winners

# response: GammaReviewResponse; manifest: loaded round-1 trial/1 record
context = ValidationContext(
    expected_stage="review_gamma",
    selected_experiment_ids=frozen_ids,
    config=config,
    reviewed_manifest=manifest,
    expected_trial_id=active_trial_id,
    expected_calibration_session_id=active_calibration_session_id,
    experiment_item_ids=experiment_to_item,
)
validated_response = validate_response(response, context)
resolved_mappings = resolve_review_winners(validated_response, context)
```

`validate_response` uses the resolved complete winner mappings internally for
next-round checks. A caller may supply `reference_mappings` in a review context,
but they must exactly equal those resolved winners, including inherited gamma,
endpoints, steps, and inversion. Stale mappings fail. No acceptance or persistence
occurs here; the future graph owns those actions after full validation succeeds.
`resolve_review_winners` is also a separate winner-validation phase that returns
copied mappings; it deliberately does not validate the next proposals. Do not
accept a complete decision solely on that phase's success.

`review_gamma` reviews physical round 1 and proposes shared endpoints for round 2;
`review_gray_levels` reviews round 2 and proposes steps for round 3;
`review_steps` reviews round 3 and has no proposals. Nonusable assessments need
issues, empty decision lists, and a null shared winner where applicable. They
need no reference mappings, but still require the reviewed manifest, trial and
calibration identities, and matching frozen selection.

Comparison uses exact binary64 numeric equality, consistent with the renderer's
float64 inputs and existing exact mapping comparisons. It does not use the
rounded printed labels, decimal rounding, or an arbitrary tolerance. Endpoint
pairs are compared together; sharing just one endpoint is allowed. Steps use
strict integers and the configuration's explicit special values, including 0.
The shared manifest-validation path checks every printed candidate, not just
winners: reference and each test label must have identical endpoints across the
four experiments. In gamma/steps rounds, tests must also retain their reference
endpoints. Gamma and steps may differ between experiments. Printed tests must
have distinct values of the active parameter and may not duplicate the reference.
Both response validation and winner resolution enforce these rules, including
for rescan responses. Invalid context types raise `TuningValidationError` before
ID hashing, sorting, or lookup.

## Existing manifest translation and integration gaps

The validator consumes the existing `trial/1` shape: `items`,
`variations[item_id]`, `mapping`, `role`, `variation_id`, `round_id`,
`tested_parameter`, `config`, `trial_id`, and `calibration_session_id`. It does
not create a replacement manifest. The supplied config must equal the manifest's
snapshot. Optional expected run/selection ID/version fields can further bind it.

The label `reference` means the sole entry with `role="reference"`. Labels
`test_1` through `test_5` mean the five `role="test"` entries in their stored
list order. Internal identifiers such as `ref` and `g0` remain unchanged. Duplicate
internal IDs, incomplete lists, or canonical IDs contradicting their assigned
labels fail validation. Future image attachment/scan presentation must use this
same translation, since the manifest has no explicit specialist candidate label.

The public helper exposes the same translation used by validation; there is no
second candidate-label algorithm:

```python
from app.print_tuning import get_candidate_correspondence

# A review-stage context, bound to the expected trial and calibration session.
identities = get_candidate_correspondence(context)
for identity in identities:
    print(identity.experiment_id, identity.item_id,
          identity.candidate_id, identity.variation_id)
```

The helper validates the loaded manifest and returns an immutable tuple of
`CandidateIdentity` records, in frozen experiment order and then
`reference`, `test_1`, ..., `test_5` order. Internal IDs and input records are
unchanged. Future image-message construction and scan-tool integration must use
this correspondence to attach the correct specialist labels. Neither integration
is implemented here.

`selection.build_selection` saves `source.experiment_id` and sanitizes `item_id`,
but `create_trial_sheet` omits `source` from manifest items. Preserve the explicit
association from the original frozen selection:

```python
experiment_to_item = {
    item["source"]["experiment_id"]: item["item_id"]
    for item in selection["items"]
}
```

If no association is supplied, only exact experiment-ID/item-ID equality is
accepted. The validator never reverses lossy sanitization. The manifest alone
cannot prove the original experiment identity when they differ. Agree with
Razmik on retaining this association at integration time and on consistent
specialist labels; no changes to the existing manifest or renderer are made here.
The existing session identity is named `calibration_session_id`; it cannot
independently prove a future separate `session_id` without an agreed association.

## Offline checks

No extra dependency is needed. Run the focused standard-library tests:

```powershell
python -m unittest discover -s tests/print_tuning -v
```

Fixtures use the existing selection packaging and actual trial-sheet generator
for all three rounds, saving only tiny arrays and temporary PDFs/manifests. They
perform no model calls, physical printing, or scans.
