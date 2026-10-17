# Report revision and fresh approval

At the **PDF print-approval prompt** (`Print this report? Type yes, or describe
what to change.`), enter the change itself instead of `yes`. For example:

```text
Add a sentence at the end asking for the reader's opinion.
```

`retry` is never approval and is not a revision. The exact revision wording,
request ID and original PDF hash are persisted in `report_print.json`. Any pending
approval is invalidated; prior attempts remain intact. Submission requests and
physical submission are blocked until generation succeeds. Resume rehydrates the
pending request from that journal, even if the saved graph projection is stale.

The executor must request `build_final_report`; the application supplies the exact
instructions and ID to that existing tool. It translates supported edits into
structured text fields or supported layout controls. It cannot change numerical
evidence, selection, tuning settings, required report content, or page policy.
No experiment or tuning round is rerun. Unsupported edits require clarification.

The tool archives the previous PDF by SHA-256 in `report_versions/` before
regeneration. It retains prior hash-bound text/layout inputs, and writes versioned
report inputs with the revision request. Cache fingerprints include the exact
revision and presentation settings. The builder validates two A4 pages and the
exact comparison image on page 2. Changed text must appear in the PDF. No-op edits,
unchanged PDF hashes, and byte-only differences without text/layout changes fail;
the old approval cannot be reused. A new hash does not prove semantic compliance.
Successful output is explicitly offered for human review and fresh approval.

Generation failure leaves the revision pending and opens a workflow recovery
prompt: `retry`, `correct` (enter replacement instructions), or `stop`. An earlier
unresolved submission is resolved through the existing human-confirmation flow
before generation or another print. Histories and uncertainty are never cleared.

## Existing completed run

Read-only inspection of `runs/20261017_134707_564060` found a request to add a reader
opinion sentence and two successful builds with different PDF hashes. The second
build's saved inputs contain "What is your opinion of these findings?"; this run
does not establish a same-byte replay of that request. The job's uncertain queue
departure has already been resolved as `printed` by the human. Ordinary resume
preserves that completion. To explicitly request a further revision:

1. Exit any old paused process with Ctrl+C; wait for exit. Never run concurrent
   instances against the folder or edit its JSON while an instance is active.
2. From the project root run:

   ```powershell
   $env:SUMATRA_PATH = "C:\Users\User\AppData\Local\SumatraPDF\SumatraPDF.exe"
   & .\.venv\Scripts\python.exe .\main.py --resume .\runs\20261017_134707_564060 --no-fake-printer --printer ir2425 --report-revision "Change the final reader question to: Which follow-up experiment would you prioritize?"
   ```

   The new option explicitly records a fresh revision for a completed run. Resume
   uses the configured paid executor; this command was not run during the fix.
   Do not use `--resume-stopped-tuning` for this completed tuning session.
3. Open the revised `report.pdf` at the new exact-PDF approval prompt. Check the
   requested sentence and both pages. Type `yes` only if you want another copy
   printed. To change it further, enter the next revision directly at that prompt.

If a revision is already pending, ordinary `--resume` is sufficient: its exact
instructions remain durable. If a newer print outcome is uncertain, resolve that
human confirmation first. Do not erase attempts or use regeneration to retry a job.

The fix itself inspected this run read-only and did not regenerate its PDF, call
paid models, contact hardware, or print.
