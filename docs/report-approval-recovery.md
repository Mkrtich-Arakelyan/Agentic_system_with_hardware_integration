# Final report approval and recovery

This records the original report/1 deadlock investigation. The application now
requires the [two-page report/2 format](two-page-report.md). The one-page PDF and
empty-attempt journal described below are historical observations, not current
validation policy or a live journal inventory. The new offline builder preserves
that original report and journal; it writes a separate bundle outside the run.

The executor's `submit_print` tool call requests the application approval
interrupt. It does not submit a print job. Once the report validates and a
structured printer reading is idle, the application exposes only `submit_print`
and uses the provider's named `tool_choice` with parallel calls disabled. The
model must still generate the call and its arguments. Two unsuccessful request
passes pause in `report_workflow_problem`; repeated status reads and the overall
report-pass limit also use workflow recovery, never a fabricated hardware fault.

The exact-PDF approval displays path, page count, printer, copies and SHA-256.
Only an explicit `yes`, `y` or `approve` at that interrupt advances submission.
`retry` reopens approval without submitting. Other text requests a revision.
Submission revalidates the PDF, figure and arguments against the persisted
pending request, writes intent before printer I/O, and never repeats a recorded
attempt. Missing outcomes require human confirmation. Ordinary resume discards
visit counters and transient approvals, while preserving the print journal.

## Run 20261016_183451_221483 / ir2425

Read-only inspection during the fix found completed tuning through round 3.
All three trial attempts have human `printed` resolutions (including the first
untracked job). `report_print.json` has `attempts: []`, `pending: null`; saved
execution evidence also contains no final-report `submit_print`. No unresolved
submission was found. Recheck these records if another process has since acted.

The existing report validates as one A4 portrait page with the exact tuned PNG.
Its SHA-256 is
`25e8291b7eaf1caa53c8d981dee46ff06fa1a9475099bfc24b738d16f6abf930`.
The run records, completed experiments, scans, mappings, final figure and PDF
were not modified by that repair. No regeneration was needed for report/1;
report/2 regeneration uses completed tuning and existing scans without retuning.

1. In the terminal running the old paused application, press **Ctrl+C** once.
   Wait for its failure/save message and for the PowerShell prompt to return.
   In PyCharm, interrupt the main run and wait until it exits. Do not answer
   `skip`, edit JSON, or start another instance while it is still running.
   Ctrl+C is currently recorded as `execution_failure`; ordinary resume handles
   that saved state. Do not kill unrelated Python processes.
2. After exit, from the repository root run:

   ```powershell
   & .\.venv\Scripts\python.exe .\main.py --resume .\runs\20261016_183451_221483 --no-fake-printer --printer ir2425
   ```

   This is the real-printer continuation command; it may call the configured
   paid executor model and read printer status. It was **not executed** during
   this repair. It reuses the saved model, completed tuning, figure and report.
   Do not use `--resume-stopped-tuning`: this session is already completed.
3. Review the PDF and the application's exact-PDF approval prompt. Only type
   `yes` when you want that displayed PDF printed on `ir2425`. Executor prose
   and earlier `retry` replies grant no printing authority.
4. If the journal instead contains an intent with a missing/uncertain outcome,
   the workflow must pause for `printed`, `not_printed`, or `stop`. Check the
   tray/job history before resolving it. Do not choose `not_printed` unless
   confirmed: another copy then requires a fresh model request and approval.

Keep `report_print.json` and `tuning_session.json` intact. Never reset history,
delete an attempt, or submit/reprint automatically to discover an outcome.
