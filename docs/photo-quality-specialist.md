# Photo quality specialist

`app.agents.photo_quality_specialist` is a tool-free specialist with four required
decision stages. It does not start a model at import time, join a graph, accept
settings, or read/write sessions. Inject an image-capable `BaseChatModel` supporting
native `with_structured_output(..., method="json_schema", include_raw=True)`.
No model name, API key, tool binding, or tool fallback is configured here.

```python
from app.agents import (
    ImageAttachment, ImageLabel, SpecialistEvidence, photo_quality_specialist,
)

# context: existing ValidationContext for propose_gamma, with complete references.
# preview_paths: explicitly supplied {(experiment_id, panel): PNG/JPEG/WebP path}.
evidence = SpecialistEvidence(
    normalization_limits=fixed_limits,
    images=[
        ImageAttachment(ImageLabel("source_preview", experiment, panel), path)
        for (experiment, panel), path in preview_paths.items()
    ],
)
response = photo_quality_specialist(llm, context, evidence)
```

The public signature is:

```python
photo_quality_specialist(
    llm: BaseChatModel,
    context: ValidationContext,
    evidence: SpecialistEvidence,
    *,
    prior_decisions: Sequence[Response] = (),
) -> Response
```

`prepare_specialist_messages(context, evidence, *, prior_decisions=())` exposes
input preparation separately. `ImageLabel` identifies image purpose, experiment,
panel, specialist candidate, internal variation, and optional page. A source
preview has experiment/panel identity and no candidate ID; a candidate scan needs
the shared candidate/variation correspondence. A `scan_page` instead identifies
an existing manifest page. Both Initial and Final source previews are required
for each frozen experiment in the initial stage. Their supplied pixels must use
the declared reference mapping and fixed normalization limits; the module cannot
infer this provenance from pixels alone.

Only explicitly supplied static PNG/JPEG/WebP files are loaded and verified.
The original bytes become actual `image_url` data message parts, alongside labels
and numerical metadata. There is no resizing, intensity rescaling, contrast
enhancement, directory search, or scan/render execution. Paths are not sent as
substitutes for images. The module has no dialogue logger and does not include
image payloads in its diagnostics or correction text. Callers should likewise
avoid dumping these multimodal messages into dialogue/session logs; any provider
or application tracing is configured outside this module.

## Stages and reference ownership

| Decision | Reviews | Returns |
| --- | --- | --- |
| `propose_gamma` | Digital previews only | Five exploratory gamma tests per experiment |
| `review_gamma` | Physical round 1 | Per-experiment gamma winners, five shared endpoint pairs |
| `review_gray_levels` | Physical round 2 | One shared endpoint winner, five steps tests per experiment |
| `review_steps` | Physical round 3 | Per-experiment steps winners, no proposals |

The first proposals are exploratory, not measured printer optima. All reviews
report best tested choices rather than proven global optima. Prompt criteria
cover useful low-value detail, meaningful gray separation, bright detail,
Initial/Final readability, measured contrast, clipping, noise, and confounders.
Colored paper is judged against measured paper/black/gray references; lack of
white pixels is not itself failure. Annotation visibility is not spot visibility.

`validate_context()` preflights configuration, selection, references and reviewed
trial before the model call. Reviews reuse `get_candidate_correspondence()`.
`validate_response()` resolves complete winners before comparing the next tests
against them. `prior_settings`, if supplied, must match the manifest reference
BEFORE that review; it is interpretive context, not a replacement for new winners.
For a review normally leave `ValidationContext.reference_mappings=None`; any
supplied value must already match the response's resolved winners. A mismatch
raises an input error without asking the model to select a different winner.
The future graph remains responsible for acceptance and persistence.

## Adapting the current scan reader

The existing implementation is still the root `scan_evaluation.py`. Its
`evaluate_scan()` returns `scan_eval/1`, not a tool contract with `status`,
`scan_quality`, or `candidates` fields. Do not assume those fields exist.

Actual fields used here:

- `trial_id`, `round_id`, `tested_parameter`: bind the supplied scan result to the
  active reviewed manifest. The specialist input checks trial and round; the
  context validator checks the manifest's tested parameter.
- `tiles[]`: `item_id`, `variation_id`, `panel`, `page_number`, `png`, and
  `measurements`. Measurements include actual `contrast`, `intended_contrast`,
  `printed_levels`, `collapsed_levels`, clipping-related paper/black shares,
  correlation, and optional focus measurements.
- `pages[]`: alignment/page-code readings, including `fit_error_mm`, `dpi`,
  `orientation`, `page_code_read`, and `page_number`.
- `printer_response`: per-page paper/black/gray-patch readings and combined values.
- `variations[]`: optional combined panel summaries keyed by item/variation, with
  role and complete mapping. They may be represented as `CandidateMeasurement`
  with `panel=None`, preserving their actual numerical fields.

A caller can adapt already loaded output as follows. This example performs no
scan processing and loads no file except the explicit tile paths during message
preparation:

```python
from app.agents import CandidateMeasurement, ImageAttachment, ImageLabel, SpecialistEvidence
from app.print_tuning import get_candidate_correspondence

identities = get_candidate_correspondence(context)
by_internal = {(c.item_id, c.variation_id): c for c in identities}
attachments, measurements = [], []
for tile in scan_result["tiles"]:
    c = by_internal[(tile["item_id"], tile["variation_id"])]
    label = ImageLabel(
        purpose="scanned_candidate", experiment_id=c.experiment_id,
        candidate_id=c.candidate_id, variation_id=c.variation_id,
        panel=tile["panel"], page_number=tile["page_number"],
    )
    attachments.append(ImageAttachment(label, tile["png"]))
    measurements.append(CandidateMeasurement(
        experiment_id=c.experiment_id, candidate_id=c.candidate_id,
        variation_id=c.variation_id, panel=tile["panel"],
        values=tile["measurements"],
    ))
evidence = SpecialistEvidence(
    normalization_limits=fixed_limits,  # original manifest items[].limits by experiment
    images=attachments, candidate_measurements=measurements,
    trial_id=scan_result["trial_id"], round_id=scan_result["round_id"],
    page_measurements=scan_result["pages"],
    printer_response=scan_result["printer_response"],
    prior_settings=previous_accepted_settings,
)
response = photo_quality_specialist(llm, context, evidence)
```

Keep `fixed_limits` associated with original experiment IDs through the explicit
experiment/item association in the frozen selection. Do not reverse lossy ID
sanitization or translate candidate labels independently. Unknown scan tiles must
fail adaptation rather than being silently discarded. The reader does not emit
a calibration-session ID; the context binds that identity through its manifest.
It does not currently emit candidate quality-warning lists or noise estimates;
leave those absent unless the application has explicitly supplied such evidence.
Do not invent them from field names. The specialist never calls this scan reader.

## Missing evidence and correction behavior

For reviews, supply every candidate panel crop, complete page images, or explicitly
describe incompleteness. `MissingImage(label, reason)` can declare individual
missing crops; `incomplete_evidence_reason` can explain a partial or fully absent
visual input. Source previews alone do not silently substitute for scans.
An unreadable attachment that is declared as present is an input error; mark it
missing instead. No missing image is discovered or regenerated automatically.
This lets the specialist return `needs_rescan`; usable scans with inadequate
printable contrast can return `insufficient_contrast`. Existing empty-list/null
rules are enforced. Neither outcome requires next-round proposal references.

There are at most two model invocations: an initial decision and ONE correction
for a schema/parsing or model-decision validation error. Correction retains the
same original images, evidence and stage, and supplies sanitized diagnostics and
the prior textual response where available. Invalid attempts accept no winners.
Two invalid attempts raise `PhotoQualitySpecialistError` with bounded diagnostics.

Missing/unreadable files, context/configuration/identity errors, and a model
profile explicitly disabling images or structured output raise
`SpecialistInputError` before invocation. Provider authentication, connection and
unsupported-capability errors propagate without a decision correction. An unknown
model profile is not proof of support: the injector must choose a compatible
image-capable model and native JSON-schema API. No live compatibility claim is
made from mocked tests.

Run offline tests with `python -m unittest discover -s tests -v`. Runner/model
selection, graph routing, image/scan adapters, approvals, and persistence remain
future integration work.
