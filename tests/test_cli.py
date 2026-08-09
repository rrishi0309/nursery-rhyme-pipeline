from typer.testing import CliRunner

from nursery.cli import app

runner = CliRunner()


def test_list_reports_nothing_when_out_dir_empty(tmp_path):
    result = runner.invoke(app, ["--out-dir", str(tmp_path), "list"])

    assert result.exit_code == 0
    assert "no videos" in result.stdout.lower()


def test_list_shows_video_ids(tmp_path, ready_manifest, cfg):
    ready_manifest.save(cfg.video_dir("v1") / "manifest.json")

    result = runner.invoke(app, ["--out-dir", str(cfg.out_dir), "list"])

    assert result.exit_code == 0
    assert "v1" in result.stdout


def test_show_prints_title_and_scene_count(ready_manifest, cfg):
    ready_manifest.save(cfg.video_dir("v1") / "manifest.json")

    result = runner.invoke(app, ["--out-dir", str(cfg.out_dir), "show", "v1"])

    assert result.exit_code == 0
    assert "Twinkle" in result.stdout
    assert "2" in result.stdout


def test_show_unknown_id_exits_nonzero(tmp_path):
    result = runner.invoke(app, ["--out-dir", str(tmp_path), "show", "nope"])

    assert result.exit_code != 0


def test_stage_runs_assemble(ready_manifest, cfg):
    ready_manifest.save(cfg.video_dir("v1") / "manifest.json")

    result = runner.invoke(app, ["--out-dir", str(cfg.out_dir), "stage", "v1", "assemble"])

    assert result.exit_code == 0
    assert (cfg.video_dir("v1") / "video" / "final.mp4").exists()


def test_clean_removes_intermediates_but_keeps_final(ready_manifest, cfg):
    ready_manifest.save(cfg.video_dir("v1") / "manifest.json")
    runner.invoke(app, ["--out-dir", str(cfg.out_dir), "stage", "v1", "assemble"])

    result = runner.invoke(app, ["--out-dir", str(cfg.out_dir), "clean", "v1"])

    assert result.exit_code == 0
    assert (cfg.video_dir("v1") / "video" / "final.mp4").exists()
    assert (cfg.video_dir("v1") / "manifest.json").exists()
    assert not (cfg.video_dir("v1") / "images").exists()
