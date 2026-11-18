import pytest

from donorshift import __version__
from donorshift.cli import main


def test_cli_demo_and_validation(tmp_path, capsys):
    path=tmp_path/'counts.csv'
    assert main(['make-demo','--output',str(path),'--donors','4','--cells-per-donor','8','--genes','24'])==0
    assert main(['validate','--input',str(path)])==0
    assert '"valid": true' in capsys.readouterr().out


def test_cli_error_code(tmp_path):
    assert main(['validate','--input',str(tmp_path/'missing.csv')])==2


def test_cli_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(['--version'])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == f'donorshift {__version__}'
