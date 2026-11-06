"""Download the original authors' pinned dataset and check its SHA-256 before saving it.

Source: Bastani et al. (2025), https://github.com/obastani/GenAICanHarmLearning
The file is saved to research_data/main_regressions/final_data.csv (not committed).
"""
from pathlib import Path
from urllib.request import urlopen
import hashlib

ROOT = Path(__file__).resolve().parent
COMMIT = '2f63dae1a01d51453826fe07ef5cf6678e339588'
URL = f'https://raw.githubusercontent.com/obastani/GenAICanHarmLearning/{COMMIT}/main_regressions/final_data.csv'
DESTINATION = ROOT / 'research_data/main_regressions/final_data.csv'
EXPECTED_SHA256 = '430055b4633eaaf87879bb38eb4b1c9425b91e1d7f529029dc1883e33d91df0a'


def main():
    with urlopen(URL, timeout=120) as response:
        content = response.read()
    if hashlib.sha256(content).hexdigest() != EXPECTED_SHA256:
        raise ValueError('The downloaded file does not match the analyzed source.')
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_bytes(content)
    print("Downloaded the original authors' data and verified its checksum.")


if __name__ == '__main__':
    main()
