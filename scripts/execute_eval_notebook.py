"""Execute saved-result notebook without external API calls; optional --results directory."""
import argparse
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from jupyter_client.kernelspec import KernelSpecManager
from app.search.cache import write_json

ROOT = Path(__file__).resolve().parents[1]


def execute(results=None, output=None):
    # Explicit interpreter; do not accidentally run a system Python kernel.
    import sys
    kernel_root = ROOT / '.cache' / 'jupyter' / 'kernels'
    write_json(kernel_root / 'rag-project' / 'kernel.json', {
        'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
        'display_name': 'RAG project', 'language': 'python',
    })
    if results:
        os.environ['RAG_RESULTS_DIR'] = str(Path(results).resolve())
    nb = nbformat.read(ROOT / 'notebook/rag_evaluation.ipynb', as_version=4)
    client = NotebookClient(nb, timeout=180, kernel_name='rag-project',
                            resources={'metadata': {'path': str(ROOT)}})
    client.create_kernel_manager()
    client.km.kernel_spec_manager = KernelSpecManager(kernel_dirs=[str(kernel_root)])
    client.execute()
    dest = Path(output or ROOT / 'notebook/rag_evaluation.ipynb')
    nbformat.write(nb, dest)
    errors = [out for cell in nb.cells for out in cell.get('outputs', []) if out.output_type == 'error']
    print(f'Executed {len(nb.cells)} cells; errors={len(errors)}; output={dest}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results')
    parser.add_argument('--output')
    args = parser.parse_args()
    execute(args.results, args.output)
