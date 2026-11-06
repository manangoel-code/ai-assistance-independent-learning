"""Run the full Python workflow in order: download, analyses, verification."""
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
STEPS = ['download_data.py', 'analysis.py', 'extended_analysis.py', 'verify_results.py']


def main():
    env = os.environ.copy()
    env['OPENBLAS_NUM_THREADS'] = '1'  # keeps linear algebra results bit-for-bit stable
    for name in STEPS:
        print('Running', name, flush=True)
        subprocess.run([sys.executable, str(ROOT / name)], cwd=ROOT, env=env, check=True)
    print('Finished. Results are in', ROOT / 'results')


if __name__ == '__main__':
    main()
