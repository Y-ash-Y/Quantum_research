"""Check public navigation and the single dependency entry point."""
from pathlib import Path
import re
from urllib.parse import unquote

ROOT=Path(__file__).resolve().parents[2]
APP=ROOT/'quantum-ai-lab'
GUIDES=['README.md','quantum-ai-lab/README.md','quantum-ai-lab/src/README.md',
        'quantum-ai-lab/scripts/README.md','quantum-ai-lab/tests/README.md',
        'quantum-ai-lab/docs/README.md','quantum-ai-lab/notebooks/README.md',
        'quantum-ai-lab/results/README.md','quantum-ai-lab/results/benchmarks/README.md',
        'output/README.md']
GUIDES += ['quantum-ai-lab/src/'+p+'/README.md' for p in
           ('bb84','classical_ml','research','adversarial','quantum_ml','sequence_ml')]


def test_single_requirements_file():
    assert (ROOT/'requirements.txt').is_file()
    assert not list(APP.glob('requirements*.txt'))
    lines=(ROOT/'requirements.txt').read_text().splitlines()
    packages=[line.split('==',1)[0].lower().replace('_','-') for line in lines
              if line.strip() and not line.startswith('#')]
    assert len(packages)==len(set(packages))


def test_public_folder_guides_and_links():
    for relative in GUIDES:
        path=ROOT/relative
        assert path.is_file(),relative
        text=path.read_text()
        for match in re.finditer(r'\]\(([^)]+)\)',text):
            target=match.group(1).split('#',1)[0]
            if not target or '://' in target or target.startswith('mailto:'):continue
            assert (path.parent/unquote(target.strip('<>'))).exists(),(relative,target)
