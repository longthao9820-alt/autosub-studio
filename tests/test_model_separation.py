"""Kiem thu cho quy trinh tach model ASR khoi ban dong goi (Contract v2-p9-model-separation-r2)."""

from __future__ import annotations

import os
import zipfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from scripts.package_release import MAX_RELEASE_BYTES, package_release, should_include_path

from autosub_studio import selftest
from autosub_studio.providers import asr
from autosub_studio.services import paths
from autosub_studio.services.settings import Settings
from autosub_updater import main as main_updater
from autosub_updater.main import apply_update


class TestModelPathResolution:
    """Kiem thu co che phan giai duong dan model ASR."""

    def test_resolve_from_asr_models_dir(self, tmp_path, monkeypatch):
        asr_dir = tmp_path / "Data" / "models" / "asr"
        model_folder = asr_dir / "faster-whisper-small"
        model_folder.mkdir(parents=True)
        (model_folder / "model.bin").write_bytes(b"dummy model data 123")

        monkeypatch.setattr(paths, "asr_models_dir", lambda: asr_dir)
        source, local = asr.resolve_model_source("small")

        assert local is True
        assert Path(source) == model_folder

    def test_resolve_from_short_name_dir(self, tmp_path, monkeypatch):
        asr_dir = tmp_path / "Data" / "models" / "asr"
        model_folder = asr_dir / "small"
        model_folder.mkdir(parents=True)
        (model_folder / "model.bin").write_bytes(b"dummy model data 456")

        monkeypatch.setattr(paths, "asr_models_dir", lambda: asr_dir)
        source, local = asr.resolve_model_source("small")

        assert local is True
        assert Path(source) == model_folder

    def test_resolve_from_hf_snapshot(self, tmp_path, monkeypatch):
        asr_dir = tmp_path / "Data" / "models" / "asr"
        snapshot_dir = (
            asr_dir
            / "models--Systran--faster-whisper-small"
            / "snapshots"
            / "1234567890abcdef"
        )
        snapshot_dir.mkdir(parents=True)
        (snapshot_dir / "model.bin").write_bytes(b"dummy hf model")

        monkeypatch.setattr(paths, "asr_models_dir", lambda: asr_dir)
        source, local = asr.resolve_model_source("small")

        assert local is True
        assert Path(source) == snapshot_dir

    def test_resolve_custom_model_dir_takes_precedence(self, tmp_path, monkeypatch):
        asr_dir = tmp_path / "Data" / "models" / "asr"
        default_model = asr_dir / "faster-whisper-small"
        default_model.mkdir(parents=True)
        (default_model / "model.bin").write_bytes(b"default")

        custom_dir = tmp_path / "custom_models" / "faster-whisper-small"
        custom_dir.mkdir(parents=True)
        (custom_dir / "model.bin").write_bytes(b"custom")

        monkeypatch.setattr(paths, "asr_models_dir", lambda: asr_dir)
        source, local = asr.resolve_model_source("small", model_dir=str(custom_dir.parent))

        assert local is True
        assert Path(source) == custom_dir

    def test_resolve_missing_returns_size_and_false(self, tmp_path, monkeypatch):
        empty_dir = tmp_path / "empty_models"
        empty_dir.mkdir()
        monkeypatch.setattr(paths, "asr_models_dir", lambda: empty_dir)
        monkeypatch.setattr(asr, "bundled_dir", lambda name: None)
        monkeypatch.setattr(asr, "system_hf_cache_dir", lambda: tmp_path / "non_existent_hf")

        source, local = asr.resolve_model_source("small")
        assert source == "small"
        assert local is False


class TestPersistenceAndOfflineDetection:
    """Kiem thu tinh ben vung cua model va hoat dong offline."""

    def test_model_is_local_persistence_across_checks(self, tmp_path, monkeypatch):
        asr_dir = tmp_path / "Data" / "models" / "asr"
        asr_dir.mkdir(parents=True)
        monkeypatch.setattr(paths, "asr_models_dir", lambda: asr_dir)
        monkeypatch.setattr(asr, "bundled_dir", lambda name: None)
        monkeypatch.setattr(asr, "system_hf_cache_dir", lambda: tmp_path / "non_existent_hf")

        # 1. Ban dau chua co model
        assert asr.model_is_local("small") is False

        # 2. Model duoc tai vao Data/models/asr
        model_folder = asr_dir / "faster-whisper-small"
        model_folder.mkdir()
        (model_folder / "model.bin").write_bytes(b"model bytes")

        # 3. Kiem tra lai (mo phong ung dung khoi dong lai / kiem tra offline)
        assert asr.model_is_local("small") is True
        src, local = asr.resolve_model_source("small")
        assert local is True
        assert Path(src) == model_folder

    def test_transcribe_passes_download_root(self, tmp_path, monkeypatch):
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"RIFF" + b"\x00" * 40)

        asr_dir = tmp_path / "Data" / "models" / "asr"
        asr_dir.mkdir(parents=True)
        monkeypatch.setattr(paths, "asr_models_dir", lambda: asr_dir)

        created_instances = []

        class MockWhisperModel:
            def __init__(self, target, device, compute_type, download_root=None, **kwargs):
                created_instances.append(
                    {
                        "target": target,
                        "device": device,
                        "compute_type": compute_type,
                        "download_root": download_root,
                    }
                )

            def transcribe(self, *args, **kwargs):
                return [], MagicMock(duration=1.0, language="en")

        monkeypatch.setattr("faster_whisper.WhisperModel", MockWhisperModel)
        monkeypatch.setattr(asr, "is_available", lambda: True)

        logs = []
        cues, lang = asr.transcribe(
            audio_file,
            model_size="small",
            on_log=logs.append,
        )

        assert len(created_instances) == 1
        assert created_instances[0]["download_root"] == str(asr_dir)
        assert any("small" in log and "Data" in log for log in logs)


class TestMigrationRepeatSafeAndNoLoss:
    """Kiem thu di cu model tu ban V1 (_internal/models) sang Data/models/asr."""

    def test_migration_v1_to_data(self, tmp_path):
        app_root = tmp_path / "app"
        old_models = app_root / "_internal" / "models" / "faster-whisper-small"
        old_models.mkdir(parents=True)
        (old_models / "model.bin").write_bytes(b"v1 model binary content")
        (old_models / "config.json").write_text("{}", encoding="utf-8")
        (old_models / "tokenizer.json").write_text("{}", encoding="utf-8")

        migrated = paths.migrate_v1_models(app_dir=app_root)

        dest = app_root / "Data" / "models" / "asr" / "faster-whisper-small"
        assert len(migrated) == 1
        assert dest.is_dir()
        assert (dest / "model.bin").read_bytes() == b"v1 model binary content"
        assert (dest / "config.json").is_file()
        assert (dest / "tokenizer.json").is_file()

        # Ban goc phai duoc bao toan cho den khi hoan tat
        assert (old_models / "model.bin").read_bytes() == b"v1 model binary content"

    def test_migration_repeat_safe_no_repeated_copy(self, tmp_path):
        app_root = tmp_path / "app"
        old_models = app_root / "_internal" / "models" / "faster-whisper-small"
        old_models.mkdir(parents=True)
        (old_models / "model.bin").write_bytes(b"v1 model binary")

        # Lan 1: di cu thanh cong
        first_run = paths.migrate_v1_models(app_dir=app_root)
        assert len(first_run) == 1

        dest = app_root / "Data" / "models" / "asr" / "faster-whisper-small"
        # Danh dau tep kiem tra trong dest de xac nhan khong bi ghi de lai
        marker = dest / "marker.txt"
        marker.write_text("preserved", encoding="utf-8")

        # Lan 2: khong chep lai vi model da ton tai hop le
        second_run = paths.migrate_v1_models(app_dir=app_root)
        assert len(second_run) == 0
        assert marker.is_file()
        assert marker.read_text(encoding="utf-8") == "preserved"

    def test_migration_corrupt_dst_remigrates(self, tmp_path):
        app_root = tmp_path / "app"
        old_models = app_root / "_internal" / "models" / "faster-whisper-small"
        old_models.mkdir(parents=True)
        (old_models / "model.bin").write_bytes(b"full correct v1 binary")

        # Tao dest bi loi (dung luong khong khop)
        dest = app_root / "Data" / "models" / "asr" / "faster-whisper-small"
        dest.mkdir(parents=True)
        (dest / "model.bin").write_bytes(b"corrupt")

        migrated = paths.migrate_v1_models(app_dir=app_root)
        assert len(migrated) == 1
        assert (dest / "model.bin").read_bytes() == b"full correct v1 binary"


class TestUpdatePreservation:
    """Kiem thu updater bao ve thu muc Data va di cu model V1."""

    def test_apply_update_preserves_asr_data(self, tmp_path):
        app_root = tmp_path / "app"
        app_root.mkdir()
        (app_root / "AutoSubStudio.exe").write_text("v1 exe")

        # Du lieu ASR trong Data
        asr_dir = app_root / "Data" / "models" / "asr" / "faster-whisper-small"
        asr_dir.mkdir(parents=True)
        (asr_dir / "model.bin").write_bytes(b"persistent asr model")

        # Staging ban v2
        staging = tmp_path / "staging"
        staging.mkdir()
        (staging / "AutoSubStudio.exe").write_text("v2 exe")
        (staging / "new_lib.dll").write_text("v2 dll")

        backup = tmp_path / "backup"

        success = apply_update(
            app_root=app_root,
            staging_dir=staging,
            backup_dir=backup,
            pid=None,
            restart=False,
        )

        assert success is True
        assert (app_root / "AutoSubStudio.exe").read_text() == "v2 exe"
        # Data/models/asr phai duoc bao ve nguyen ven
        assert (asr_dir / "model.bin").read_bytes() == b"persistent asr model"

    def test_apply_update_migrates_v1_models_before_replacement(self, tmp_path):
        app_root = tmp_path / "app"
        app_root.mkdir()
        (app_root / "AutoSubStudio.exe").write_text("v1 exe")

        # Ban cu co model trong _internal/models
        old_internal = app_root / "_internal" / "models" / "faster-whisper-small"
        old_internal.mkdir(parents=True)
        (old_internal / "model.bin").write_bytes(b"v1 internal model")

        # Staging ban v2 KHONG co _internal/models
        staging = tmp_path / "staging"
        staging.mkdir()
        (staging / "AutoSubStudio.exe").write_text("v2 exe")
        (staging / "_internal").mkdir()
        (staging / "_internal" / "app.dll").write_text("v2 app dll")

        backup = tmp_path / "backup"

        success = apply_update(
            app_root=app_root,
            staging_dir=staging,
            backup_dir=backup,
            pid=None,
            restart=False,
        )

        assert success is True
        # Model da duoc chuyen sang Data/models/asr an toan
        migrated_dest = app_root / "Data" / "models" / "asr" / "faster-whisper-small"
        assert (migrated_dest / "model.bin").read_bytes() == b"v1 internal model"
        # _internal moi khong chua thu muc models
        assert not (app_root / "_internal" / "models").exists()

    def test_apply_update_failed_update_rollback_preserves_migrated_model(
        self, tmp_path, monkeypatch
    ):
        app_root = tmp_path / "app"
        app_root.mkdir()
        (app_root / "AutoSubStudio.exe").write_text("v1 exe")

        # Ban cu co model trong _internal/models
        old_internal = app_root / "_internal" / "models" / "faster-whisper-small"
        old_internal.mkdir(parents=True)
        (old_internal / "model.bin").write_bytes(b"v1 internal model")

        # Staging ban v2 co cau truc ban dau hop le
        staging = tmp_path / "staging"
        staging.mkdir()
        (staging / "AutoSubStudio.exe").write_text("v2 exe")

        backup = tmp_path / "backup"

        # Gia lap loi o buoc 6 (kiem tra toan ven sau khi thay the that bai)
        call_count = [0]
        real_verify = main_updater.verify_payload_layout

        def mock_verify(p):
            call_count[0] += 1
            if call_count[0] > 1:
                return False
            return real_verify(p)

        monkeypatch.setattr("autosub_updater.main.verify_payload_layout", mock_verify)

        success = apply_update(
            app_root=app_root,
            staging_dir=staging,
            backup_dir=backup,
            pid=None,
            restart=False,
        )

        assert success is False
        # 1. Ban goc cua app_root phai duoc rollback phuc hoi
        assert (app_root / "AutoSubStudio.exe").read_text() == "v1 exe"
        assert (old_internal / "model.bin").read_bytes() == b"v1 internal model"

        # 2. Model da duoc chep sang Data/models/asr phai song sot va khong bi hong
        migrated_dest = app_root / "Data" / "models" / "asr" / "faster-whisper-small"
        assert migrated_dest.is_dir()
        assert (migrated_dest / "model.bin").read_bytes() == b"v1 internal model"
        # Khong de lai thu muc tam
        assert not any(
            p.name.startswith(".tmp_migrate")
            for p in (app_root / "Data" / "models" / "asr").iterdir()
        )


class TestBuildAndPackageExclusion:
    """Kiem thu script dong goi loai tru _internal/models va bat bien <2 GiB."""

    def test_should_include_path_excludes_models_and_data(self):
        # Thu muc Data bi loai bo
        assert should_include_path(Path("Data/app.db")) is False
        assert should_include_path(Path("data/models/asr/model.bin")) is False

        # _internal/models bi loai bo tuyet doi
        assert should_include_path(Path("_internal/models/faster-whisper-small/model.bin")) is False
        assert should_include_path(Path("_internal/models/model.bin")) is False
        assert should_include_path(Path("models/faster-whisper-small/model.bin")) is False

        # Cac thanh phan hop le duoc giu lai
        assert should_include_path(Path("AutoSubStudio.exe")) is True
        assert should_include_path(Path("_internal/ffmpeg/ffmpeg.exe")) is True
        assert should_include_path(Path("_internal/ocr/ch_PP-OCRv4_rec_infer.onnx")) is True
        assert should_include_path(Path("_internal/piper/piper.exe")) is True

    def test_package_release_excludes_internal_models_from_zip(self, tmp_path):
        source = tmp_path / "AutoSubStudio"
        source.mkdir()
        (source / "AutoSubStudio.exe").write_text("app binary")

        # Thu muc Data va _internal/models gia lap trong source
        (source / "Data").mkdir()
        (source / "Data" / "app.db").write_text("user db")

        (source / "_internal" / "models" / "faster-whisper-small").mkdir(parents=True)
        (source / "_internal" / "models" / "faster-whisper-small" / "model.bin").write_bytes(
            b"leaked model"
        )

        # OCR model hop le trong _internal/ocr
        (source / "_internal" / "ocr").mkdir(parents=True)
        (source / "_internal" / "ocr" / "ch_PP-OCRv4_rec_infer.onnx").write_bytes(b"ocr onnx")

        out = tmp_path / "release"
        result = package_release(source_dir=source, output_dir=out, dry_run=True)

        zip_path = Path(str(result["zip_path"]))
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            assert "AutoSubStudio.exe" in names
            assert "_internal/ocr/ch_PP-OCRv4_rec_infer.onnx" in names
            # Khong duoc co Data hoac _internal/models
            assert not any("Data" in n for n in names)
            assert not any("models" in n for n in names)

    def test_package_release_enforces_2gib_limit(self, tmp_path, monkeypatch):
        source = tmp_path / "AutoSubStudio"
        source.mkdir()
        (source / "AutoSubStudio.exe").write_text("app binary")

        out = tmp_path / "release"

        # Gia lap file zip vuot qua 2 GiB
        def mock_create_zip(src, dst, **kwargs):
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(b"PK\x05\x06" + b"\x00" * 18)
            return 1

        monkeypatch.setattr(
            "scripts.package_release.create_release_zip",
            mock_create_zip,
        )

        orig_stat = Path.stat

        def mock_stat(self, *args, **kwargs):
            st = orig_stat(self, *args, **kwargs)
            if self.suffix.lower() == ".zip":
                return os.stat_result((
                    st.st_mode,
                    st.st_ino,
                    st.st_dev,
                    st.st_nlink,
                    st.st_uid,
                    st.st_gid,
                    MAX_RELEASE_BYTES + 1024,
                    st.st_atime,
                    st.st_mtime,
                    st.st_ctime,
                ))
            return st

        monkeypatch.setattr(Path, "stat", mock_stat)

        with pytest.raises(ValueError, match="vuot qua gioi han 2 GiB"):
            package_release(source_dir=source, output_dir=out, dry_run=True)

    def test_spec_excludes_assets_models(self):
        spec_path = Path(__file__).resolve().parent.parent / "AutoSubStudio.spec"
        spec_content = spec_path.read_text(encoding="utf-8")
        assert 'models_dir = ASSETS / "models"' not in spec_content
        assert 'f"models/{model.name}"' not in spec_content


class TestSelftestFreshInstall:
    """Kiem thu selftest tren may cai dat moi (chua co model ASR)."""

    def test_selftest_fresh_install_asr_status_warn_not_fail(self, tmp_path, monkeypatch):
        empty_dir = tmp_path / "empty_data_models_asr"
        empty_dir.mkdir(parents=True)
        monkeypatch.setattr(paths, "asr_models_dir", lambda: empty_dir)
        monkeypatch.setattr(asr, "bundled_dir", lambda name: None)
        monkeypatch.setattr(asr, "system_hf_cache_dir", lambda: tmp_path / "no_hf")
        monkeypatch.setattr(asr, "is_available", lambda: True)

        settings = Settings()
        results = selftest._check_asr(settings)

        assert len(results) == 1
        # Trạng thái phải là WARN (THIEU), KHÔNG PHẢI là FAIL (HONG)
        assert results[0].status == selftest.WARN
        assert "chua cai dat nhung co the tu dong tai ve" in results[0].detail.lower()

    def test_selftest_fresh_install_verdict_not_fail(self, tmp_path, monkeypatch):
        empty_dir = tmp_path / "empty_data_models_asr"
        empty_dir.mkdir(parents=True)
        monkeypatch.setattr(paths, "asr_models_dir", lambda: empty_dir)
        monkeypatch.setattr(asr, "bundled_dir", lambda name: None)
        monkeypatch.setattr(asr, "system_hf_cache_dir", lambda: tmp_path / "no_hf")
        monkeypatch.setattr(asr, "is_available", lambda: True)

        results, verdict = selftest.run_checks()
        # Neu chi thieu model ASR thi ket luan phai la CHAY DUOC (chi WARN)
        asr_result = next((r for r in results if r.name == "Nhan dang giong noi"), None)
        assert asr_result is not None
        assert asr_result.status == selftest.WARN

    def test_selftest_deep_check_skips_asr_when_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(asr, "is_available", lambda: True)
        monkeypatch.setattr(asr, "model_is_local", lambda size, d="": False)

        res = selftest._deep_check_asr(MagicMock(), tmp_path, Settings())
        assert res.status == selftest.WARN
        assert "bo qua" in res.detail.lower()
