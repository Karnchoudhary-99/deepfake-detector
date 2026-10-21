import json
import shutil

from deepfake_detector.cli import main


def test_cli_classify_text(sample_images, capsys):
    assert main(["classify", str(sample_images["fake"])]) == 0
    out = capsys.readouterr().out
    assert "verdict: likely fake" in out
    assert "evidence:" in out
    assert "high-frequency energy" in out


def test_cli_classify_json(sample_images, capsys):
    assert main(["classify", str(sample_images["real"]), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["verdict"]["label"] == "likely real"
    assert 0.0 <= payload["verdict"]["score"] <= 1.0
    assert payload["report"]["frame_count"] == 1
    assert payload["verdict"]["explanation"]


def test_cli_classify_missing_file(capsys):
    assert main(["classify", "/tmp/does-not-exist-xyz.png"]) == 1
    assert "error" in capsys.readouterr().err


def test_cli_calibrate_then_classify_with_profile(sample_images, tmp_path, capsys):
    real_dir, fake_dir = tmp_path / "real", tmp_path / "fake"
    real_dir.mkdir()
    fake_dir.mkdir()
    shutil.copy(sample_images["real"], real_dir / "real.png")
    shutil.copy(sample_images["fake"], fake_dir / "fake.png")
    profile = tmp_path / "profile.json"
    assert main(["calibrate", str(real_dir), str(fake_dir),
                 "--out", str(profile)]) == 0
    out = capsys.readouterr().out
    assert "fitted profile from 1 real + 1 fake samples" in out
    saved = json.loads(profile.read_text())
    assert saved["version"] == 1
    assert any(s["key"] == "high_freq_ratio" for s in saved["specs"])
    # A profile fitted on these samples must still separate them.
    assert main(["classify", str(sample_images["fake"]), "--json",
                 "--profile", str(profile)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["verdict"]["label"] == "likely fake"


def test_cli_calibrate_empty_dirs_error(tmp_path, capsys):
    real_dir, fake_dir = tmp_path / "r", tmp_path / "f"
    real_dir.mkdir()
    fake_dir.mkdir()
    assert main(["calibrate", str(real_dir), str(fake_dir),
                 "--out", str(tmp_path / "p.json")]) == 1
    assert "error" in capsys.readouterr().err


def test_cli_classify_bad_profile(sample_images, tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"version": 99, "specs": []}))
    assert main(["classify", str(sample_images["real"]),
                 "--profile", str(bad)]) == 1
    assert "error" in capsys.readouterr().err
