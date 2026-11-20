#!/usr/bin/env bash
# Run the daily pipeline on your own machine (Mac/Linux). Schedule it with cron in your LOCAL time zone,
# at least ~2 hours after the 4 pm New York close. Example: weekdays at 5 pm Arizona time (7-8 pm New York):
#   crontab -e
#   0 17 * * 1-5  /path/to/repo/scripts/run_daily.sh
# The computer must be awake at that time; use GitHub Actions (.github/workflows/daily_pipeline.yml) if it is not.
# On Windows, point Task Scheduler at:  <repo>\.venv\Scripts\python.exe -m src.pipeline  (start in: <repo>)
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi
python -m src.pipeline "$@" >> logs/cron.log 2>&1
