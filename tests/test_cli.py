"""Installed DynaMath entry point works without the former package or corpus."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'models/dynamath/001a/five-1tick.pt'


def run(args, directory, input=None):
    result = subprocess.run([sys.executable, '-m', 'dynamath', *args], cwd=directory,
                            input=input, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_installed_cli_generates_from_saved_model_outside_checkout(tmp_path):
    # No corpus, package files or relative model path are available in this cwd.
    result = json.loads(run(['generate', '--model', str(MODEL), '--question', '안녕하세요',
                             '--state-out', str(tmp_path / 'state.pt')], tmp_path))
    assert result['text'] == '알 어예요.'
    assert result['eos']
    state = json.loads(run(['tick', '--state-in', str(tmp_path / 'state.pt'), '--count', '1'], tmp_path))
    assert state['ticks_before'] == result['ticks']
    assert state['ticks_after'] == result['ticks'] + 1


def test_installed_chat_retains_state_and_exposes_new_cli(tmp_path):
    help_text = run(['--help'], tmp_path)
    assert 'generate' in help_text and 'train' in help_text and 'tick' in help_text
    output = run(['chat', '--model', str(MODEL), '--state-out', str(tmp_path / 'chat.pt')],
                 tmp_path, input='안녕하세요\n고마워\n/quit\n')
    assert 'DynaMath 001a' in output
    assert '"question": "안녕하세요"' in output and '"question": "고마워"' in output
    state = json.loads(run(['tick', '--state-in', str(tmp_path / 'chat.pt'), '--count', '1'], tmp_path))
    assert state['ticks_before'] > 26
