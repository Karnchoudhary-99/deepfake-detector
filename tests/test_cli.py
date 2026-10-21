import json

import pytest

from deepfake_detector.cli import main


def test_cli_text_report(sample_images, capsys):
    assert main(["analyze", str(sample_images["fake"])]) == 0
    out = capsys.readouterr().out
    assert "frames analysed: 1" in out
    assert "high_freq_ratio" in out
    assert "method=skin" in out


def test_cli_json_report(sample_images, capsys):
    assert main(["analyze", str(sample_images["real"]), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["frame_count"] == 1
    assert report["frames"][0]["face"]["method"] == "skin"
    assert "high_freq_ratio" in report["averages"]


def test_cli_missing_file_errors(capsys):
    assert main(["analyze", "/tmp/does-not-exist-xyz.png"]) == 1
    assert "error" in capsys.readouterr().err


def test_cli_requires_subcommand():
    with pytest.raises(SystemExit):
        main([])
