# Windows trial-print executable

The existing connector reads `SUMATRA_PATH` from its environment. The terminal
runner loads the project `.env` and passes its environment to the printer MCP
process. Shell or IDE environment variables take precedence over `.env`.
Set the actual installed executable, not an assumed Program Files location:

```dotenv
SUMATRA_PATH=C:\Users\User\AppData\Local\SumatraPDF\SumatraPDF.exe
```

Check `%LOCALAPPDATA%\SumatraPDF`, Program Files, `Get-Command SumatraPDF.exe`,
Windows App Paths, and uninstall registration when locating an installation.
If it is absent, install SumatraPDF and set this same supported setting.
Startup checks the configured executable without launching it, before models,
approval, run creation or recovery writes. Submission rechecks before queue I/O.
This checks local executable availability, not printer readiness or driver duplex
support. There is no alternative trial-print implementation or automatic fallback.

## Recovery for 20261017_134707_564060

Inspection found SumatraPDF 3.6.1 in the user-local location above, also confirmed
by HKCU uninstall registration. PATH and checked App Paths did not locate it;
neither Program Files installation existed. The stopped round-1 session retains
one failed attempt (`command not found`), no job ID, and no human resolution.
No unknown submission outcome was recorded. All run files remain unchanged.

1. Stop any paused instance with Ctrl+C; wait for the PowerShell prompt. Do not
   edit JSON or run two instances against the same run.
2. In the project root, explicitly override any stale shell/IDE path and resume:

   ```powershell
   $env:SUMATRA_PATH = "C:\Users\User\AppData\Local\SumatraPDF\SumatraPDF.exe"
   & .\.venv\Scripts\python.exe .\main.py --resume .\runs\20261017_134707_564060 --resume-stopped-tuning --no-fake-printer --printer ir2425
   ```

   The stopped-tuning flag is necessary for this recorded human stop. The existing
   recovery backs up JSON and reopens durable progress; it does not clear attempts,
   reset limits, regenerate a valid trial, or print. Resume may call paid models.
3. At the failed-print recovery prompt, choose `reprint` only when ready to request
   another attempt. Review the existing exact PDF, printer and duplex settings;
   only explicit approval at the new approval prompt authorizes submission.
   Existing reprint limits still apply. Do not regenerate or delete history.
4. If the run has since acquired an uncertain/missing outcome, check the tray and
   job history first and resolve its human confirmation prompt. Never blindly retry.

The fix itself did not launch SumatraPDF, contact a printer, or resume this run.
