"""Kiem thu he thong cap nhat tu dong (Contract v2-p7-updater-r2)."""

from __future__ import annotations

import io
import json
import urllib.error
import zipfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PySide6.QtWidgets import QMessageBox
from scripts.package_release import package_release

from autosub_studio.services.settings import Settings
from autosub_studio.services.updater import (
    DEFAULT_REPO,
    ReleaseInfo,
    SecurityError,
    UpdateCancelledError,
    UpdateVerificationError,
    compare_versions,
    download_release_asset,
    extract_update_archive,
    fetch_checksum_from_url,
    fetch_latest_release,
    is_version_newer,
    parse_version_tuple,
    resolve_payload_root,
    verify_package_layout,
)
from autosub_studio.ui.panels import SettingsPanel
from autosub_studio.version import APP_VERSION
from autosub_updater.main import (
    apply_update,
    resolve_payload_dir,
    wait_for_pid,
)


class TestVersionComparison:
    """Kiem thu cac ham so sanh phien ban."""

    def test_parse_version_tuple(self):
        assert parse_version_tuple("2.0.0") == (2, 0, 0)
        assert parse_version_tuple("v2.1.3") == (2, 1, 3)
        assert parse_version_tuple("V3.0.0-rc1") == (3, 0, 0)
        assert parse_version_tuple("1.2") == (1, 2, 0)

    def test_compare_versions(self):
        assert compare_versions("2.0.0", "2.0.0") == 0
        assert compare_versions("2.0.1", "2.0.0") == 1
        assert compare_versions("1.9.9", "2.0.0") == -1
        assert compare_versions("v2.1.0", "2.0.9") == 1
        assert compare_versions("3.0.0-beta", "2.9.9") == 1

    def test_is_version_newer(self):
        assert not is_version_newer("2.0.0", "2.0.0")
        assert is_version_newer("2.0.1", "2.0.0")
        assert not is_version_newer("1.9.9", "2.0.0")
        assert is_version_newer("3.0.0", "2.0.0")


class TestReleaseMetadata:
    """Kiem thu lay thong tin release tu GitHub API."""

    def test_fetch_latest_release_success(self, monkeypatch):
        fake_api_response = {
            "tag_name": "v2.2.0",
            "name": "AutoSub Studio v2.2.0",
            "body": "Nang cap tinh nang moi\nFix bugs",
            "draft": False,
            "prerelease": False,
            "published_at": "2026-09-20T10:00:00Z",
            "assets": [
                {
                    "name": "AutoSubStudio-v2.2.0-windows-x64.zip",
                    "browser_download_url": "https://github.com/test/download.zip",
                    "size": 104857600,
                },
                {
                    "name": "AutoSubStudio-v2.2.0-windows-x64.zip.sha256",
                    "browser_download_url": "https://github.com/test/download.zip.sha256",
                    "size": 64,
                },
            ],
        }

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

            def read(self):
                return json.dumps(fake_api_response).encode("utf-8")

        monkeypatch.setattr(
            "urllib.request.urlopen", lambda req, timeout=20.0: FakeResponse()
        )

        rel = fetch_latest_release(repo=DEFAULT_REPO)
        assert rel is not None
        assert rel.version == "2.2.0"
        assert rel.tag_name == "v2.2.0"
        assert rel.asset_name == "AutoSubStudio-v2.2.0-windows-x64.zip"
        assert rel.asset_url == "https://github.com/test/download.zip"
        assert rel.asset_size == 104857600
        assert rel.sha256_url == "https://github.com/test/download.zip.sha256"
        assert rel.is_newer is True
        assert "Nang cap tinh nang moi" in rel.changelog

    def test_fetch_latest_release_prerelease_ignored(self, monkeypatch):
        fake_api_response = {
            "tag_name": "v2.2.0-beta",
            "draft": False,
            "prerelease": True,  # Prerelease bi loai bo tren Stable channel
            "assets": [
                {
                    "name": "AutoSubStudio-v2.2.0-beta.zip",
                    "browser_download_url": "https://github.com/test/beta.zip",
                    "size": 1000,
                }
            ],
        }

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

            def read(self):
                return json.dumps(fake_api_response).encode("utf-8")

        monkeypatch.setattr(
            "urllib.request.urlopen", lambda req, timeout=20.0: FakeResponse()
        )

        rel = fetch_latest_release(repo=DEFAULT_REPO)
        assert rel is None

    def test_fetch_latest_release_network_error(self, monkeypatch):
        def fake_error(req, timeout=20.0):
            raise urllib.error.URLError("Network unreachable")

        monkeypatch.setattr("urllib.request.urlopen", fake_error)
        assert fetch_latest_release(repo=DEFAULT_REPO) is None

    def test_fetch_checksum_from_url(self, monkeypatch):
        raw_sha = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"
        content = f"{raw_sha}  AutoSubStudio.zip\n"

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

            def read(self):
                return content.encode("utf-8")

        monkeypatch.setattr(
            "urllib.request.urlopen", lambda req, timeout=20.0: FakeResponse()
        )
        sha = fetch_checksum_from_url("https://example.com/file.sha256", "AutoSubStudio.zip")
        assert sha == raw_sha


class TestDownloadAndVerification:
    """Kiem thu tai ve tep tam .part, tien do, huy va xac thuc SHA256."""

    def test_download_success_with_sha256(self, tmp_path, monkeypatch):
        data = b"AutoSubStudio release payload contents"
        import hashlib

        expected_hash = hashlib.sha256(data).hexdigest()

        class FakeDownloadResponse:
            def __init__(self):
                self.headers = {"Content-Length": str(len(data))}
                self.stream = io.BytesIO(data)

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

            def read(self, chunk_size):
                return self.stream.read(chunk_size)

        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda req, timeout=60.0: FakeDownloadResponse(),
        )

        rel = ReleaseInfo(
            version="2.1.0",
            tag_name="v2.1.0",
            title="Test",
            changelog="Notes",
            published_at="",
            asset_name="update.zip",
            asset_url="https://fake.url/update.zip",
            asset_size=len(data),
            sha256=expected_hash,
        )

        progress_calls = []

        def on_progress(cur, tot):
            progress_calls.append((cur, tot))

        zip_path = download_release_asset(
            rel,
            tmp_path,
            on_progress=on_progress,
        )

        assert zip_path.is_file()
        assert zip_path.name == "update.zip"
        assert not (tmp_path / "update.zip.part").exists()
        assert zip_path.read_bytes() == data
        assert len(progress_calls) > 0

    def test_download_sha256_mismatch_cleans_up_part(self, tmp_path, monkeypatch):
        data = b"Corrupted data"

        class FakeDownloadResponse:
            def __init__(self):
                self.headers = {"Content-Length": str(len(data))}
                self.stream = io.BytesIO(data)

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

            def read(self, chunk_size):
                return self.stream.read(chunk_size)

        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda req, timeout=60.0: FakeDownloadResponse(),
        )

        rel = ReleaseInfo(
            version="2.1.0",
            tag_name="v2.1.0",
            title="Test",
            changelog="Notes",
            published_at="",
            asset_name="update.zip",
            asset_url="https://fake.url/update.zip",
            asset_size=len(data),
            sha256="0000000000000000000000000000000000000000000000000000000000000000",
        )

        with pytest.raises(UpdateVerificationError):
            download_release_asset(rel, tmp_path)

        assert not (tmp_path / "update.zip").exists()
        assert not (tmp_path / "update.zip.part").exists()

    def test_download_cancellation(self, tmp_path, monkeypatch):
        data = b"Some data" * 1000

        class FakeSlowStream:
            def __init__(self):
                self.headers = {"Content-Length": str(len(data))}
                self.stream = io.BytesIO(data)

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

            def read(self, chunk_size):
                return self.stream.read(10)

        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda req, timeout=60.0: FakeSlowStream(),
        )

        rel = ReleaseInfo(
            version="2.1.0",
            tag_name="v2.1.0",
            title="Test",
            changelog="Notes",
            published_at="",
            asset_name="update.zip",
            asset_url="https://fake.url/update.zip",
            asset_size=len(data),
            sha256="a" * 64,
        )

        call_count = [0]

        def cancel_flag():
            call_count[0] += 1
            return call_count[0] >= 2

        with pytest.raises(UpdateCancelledError):
            download_release_asset(
                rel,
                tmp_path,
                cancel_flag=cancel_flag,
                chunk_size=10,
            )

        assert not (tmp_path / "update.zip").exists()
        assert not (tmp_path / "update.zip.part").exists()

    def test_download_missing_sha256_raises_error_and_leaves_no_files(self, tmp_path):
        rel = ReleaseInfo(
            version="2.1.0",
            tag_name="v2.1.0",
            title="Test",
            changelog="Notes",
            published_at="",
            asset_name="update.zip",
            asset_url="https://fake.url/update.zip",
            asset_size=100,
            sha256="",
            sha256_url="",
        )

        with pytest.raises(UpdateVerificationError):
            download_release_asset(rel, tmp_path)

        assert not (tmp_path / "update.zip").exists()
        assert not (tmp_path / "update.zip.part").exists()

    def test_download_invalid_format_sha256_raises_error_and_leaves_no_files(self, tmp_path):
        rel = ReleaseInfo(
            version="2.1.0",
            tag_name="v2.1.0",
            title="Test",
            changelog="Notes",
            published_at="",
            asset_name="update.zip",
            asset_url="https://fake.url/update.zip",
            asset_size=100,
            sha256="not_a_valid_64_hex_checksum",
            sha256_url="",
        )

        with pytest.raises(UpdateVerificationError):
            download_release_asset(rel, tmp_path)

        assert not (tmp_path / "update.zip").exists()
        assert not (tmp_path / "update.zip.part").exists()


class TestZipSafetyAndExtraction:
    """Kiem thu giai nen an toan chong zip traversal va kiem tra layout."""

    def test_safe_zip_extraction(self, tmp_path):
        zip_file = tmp_path / "test.zip"
        staging_dir = tmp_path / "staging"

        with zipfile.ZipFile(zip_file, "w") as zf:
            zf.writestr("AutoSubStudio.exe", b"fake exe")
            zf.writestr("README.md", b"fake readme")

        payload = extract_update_archive(zip_file, staging_dir)
        assert (payload / "AutoSubStudio.exe").is_file()
        assert (payload / "README.md").is_file()
        assert verify_package_layout(payload) is True

    def test_extract_cleans_stale_content_preserves_zip_and_parent(self, tmp_path):
        parent_dir = tmp_path / "Data"
        parent_dir.mkdir()
        (parent_dir / "user_db.sqlite").write_text("protected db")

        staging_dir = parent_dir / "staging" / "update_2.1.0"
        staging_dir.mkdir(parents=True)

        zip_file = staging_dir / "update.zip"
        with zipfile.ZipFile(zip_file, "w") as zf:
            zf.writestr("AutoSubStudio.exe", b"new exe")
            zf.writestr("new_lib.dll", b"new dll")

        # Tao cac tep/thu muc rac con sot lai tu lan giai nen truoc
        stale_dir = staging_dir / "old_extract"
        stale_dir.mkdir()
        (stale_dir / "old_file.txt").write_text("stale")
        (staging_dir / "stale_binary.tmp").write_text("stale binary")

        payload = extract_update_archive(zip_file, staging_dir)

        # Tep zip duoc bao ve nguyen ven
        assert zip_file.is_file()
        # Tep rac trong staging bi don dep sach se
        assert not stale_dir.exists()
        assert not (staging_dir / "stale_binary.tmp").exists()
        # Tep va thu muc Data cha khong bi anh huong
        assert (parent_dir / "user_db.sqlite").read_text() == "protected db"
        # Noi dung moi duoc giai nen day du
        assert (payload / "AutoSubStudio.exe").is_file()
        assert (payload / "new_lib.dll").is_file()

    def test_zip_traversal_attack_blocked(self, tmp_path):
        zip_file = tmp_path / "evil.zip"
        staging_dir = tmp_path / "staging"

        with zipfile.ZipFile(zip_file, "w") as zf:
            zf.writestr("../evil.txt", b"evil")

        with pytest.raises(SecurityError):
            extract_update_archive(zip_file, staging_dir)

    def test_zip_absolute_path_blocked(self, tmp_path):
        zip_file = tmp_path / "evil_abs.zip"
        staging_dir = tmp_path / "staging"

        with zipfile.ZipFile(zip_file, "w") as zf:
            zf.writestr("/evil.txt", b"evil")

        with pytest.raises(SecurityError):
            extract_update_archive(zip_file, staging_dir)

    def test_nested_payload_root_resolution(self, tmp_path):
        staging_dir = tmp_path / "staging"
        inner = staging_dir / "AutoSubStudio"
        inner.mkdir(parents=True)
        (inner / "AutoSubStudio.exe").write_text("binary")

        assert resolve_payload_root(staging_dir) == inner
        assert resolve_payload_dir(staging_dir) == inner


class TestTransactionalApplyAndRollback:
    """Kiem thu quy trinh thay the phien ban: transactional, rollback, bao ve Data."""

    def test_apply_success_and_data_preservation(self, tmp_path):
        app_root = tmp_path / "app"
        app_root.mkdir()
        (app_root / "AutoSubStudio.exe").write_text("v1 exe")
        (app_root / "config.txt").write_text("v1 config")

        # Thu muc Data chua du lieu nguoi dung phai duoc bao ve 100%
        data_dir = app_root / "Data"
        data_dir.mkdir()
        (data_dir / "app.db").write_text("user db")
        (data_dir / "projects").mkdir()
        (data_dir / "projects" / "p1.json").write_text("user project 1")
        (data_dir / "models").mkdir()
        (data_dir / "models" / "piper.onnx").write_text("user model")

        # Staging chua ban v2
        staging = tmp_path / "staging"
        staging.mkdir()
        (staging / "AutoSubStudio.exe").write_text("v2 exe")
        (staging / "config.txt").write_text("v2 config")
        (staging / "new_feature.dll").write_text("v2 dll")

        backup = tmp_path / "backup"

        success = apply_update(
            app_root=app_root,
            staging_dir=staging,
            backup_dir=backup,
            pid=None,
            restart=False,
        )

        assert success is True
        # Kiem tra file ung dung da duoc cap nhat len v2
        assert (app_root / "AutoSubStudio.exe").read_text() == "v2 exe"
        assert (app_root / "config.txt").read_text() == "v2 config"
        assert (app_root / "new_feature.dll").read_text() == "v2 dll"

        # Kiem tra thu muc Data nguoi dung duoc bao ve nguyen ven
        assert (data_dir / "app.db").read_text() == "user db"
        assert (data_dir / "projects" / "p1.json").read_text("utf-8") == "user project 1"
        assert (data_dir / "models" / "piper.onnx").read_text("utf-8") == "user model"

        # Kiem tra backup da duoc tao va luu lai
        assert (backup / "AutoSubStudio.exe").read_text() == "v1 exe"
        assert (backup / "config.txt").read_text() == "v1 config"

    def test_apply_failure_triggers_rollback(self, tmp_path, monkeypatch):
        app_root = tmp_path / "app"
        app_root.mkdir()
        (app_root / "AutoSubStudio.exe").write_text("v1 exe")
        (app_root / "keep_me.txt").write_text("original text")

        staging = tmp_path / "staging"
        staging.mkdir()
        (staging / "AutoSubStudio.exe").write_text("v2 exe")
        (staging / "keep_me.txt").write_text("corrupted text")

        backup = tmp_path / "backup"

        # Gia lap loi verification sau khi copy
        monkeypatch.setattr(
            "autosub_updater.main.verify_payload_layout",
            lambda p: p != app_root,
        )

        success = apply_update(
            app_root=app_root,
            staging_dir=staging,
            backup_dir=backup,
            pid=None,
            restart=False,
        )

        assert success is False
        # Rollback da phuc hoi nguyen trang ban v1
        assert (app_root / "AutoSubStudio.exe").read_text() == "v1 exe"
        assert (app_root / "keep_me.txt").read_text() == "original text"

    def test_restart_command_invocation(self, tmp_path, monkeypatch):
        app_root = tmp_path / "app"
        app_root.mkdir()
        (app_root / "AutoSubStudio.exe").write_text("v2 exe")

        staging = tmp_path / "staging"
        staging.mkdir()
        (staging / "AutoSubStudio.exe").write_text("v2 exe")

        backup = tmp_path / "backup"

        spawned_cmds = []

        def fake_popen(cmd, **kwargs):
            spawned_cmds.append(cmd)
            mock = MagicMock()
            return mock

        monkeypatch.setattr("subprocess.Popen", fake_popen)

        success = apply_update(
            app_root=app_root,
            staging_dir=staging,
            backup_dir=backup,
            pid=None,
            restart=True,
            restart_cmd=["dummy_restart.exe", "--arg1"],
        )

        assert success is True
        assert len(spawned_cmds) == 1
        assert spawned_cmds[0] == ["dummy_restart.exe", "--arg1"]

    def test_wait_for_pid(self):
        # PID khong hop le hoac <= 0 thi tra ve True ngay
        assert wait_for_pid(0) is True
        assert wait_for_pid(-1) is True
        # PID khong ton tai (9999999) tra ve True vi da thoat
        assert wait_for_pid(9999999, timeout=1.0) is True


class TestUIElementsAndSignals:
    """Kiem thu cac thanh phan UI SettingsPanel va MainWindow update section."""

    def test_settings_panel_update_widgets(self, qapp):
        panel = SettingsPanel()
        assert hasattr(panel, "lbl_current_version")
        assert hasattr(panel, "lbl_channel")
        assert hasattr(panel, "auto_check_update")
        assert hasattr(panel, "btn_check_update")
        assert hasattr(panel, "lbl_update_status")
        assert hasattr(panel, "lbl_new_version")
        assert hasattr(panel, "lbl_update_size")
        assert hasattr(panel, "txt_changelog")
        assert hasattr(panel, "btn_update_now")
        assert hasattr(panel, "update_progress")

        assert APP_VERSION in panel.lbl_current_version.text()
        assert "Stable" in panel.lbl_channel.text()

    def test_settings_panel_load_and_apply(self, qapp):
        panel = SettingsPanel()
        s = Settings()
        s.auto_check_update = False
        panel.load(s, api_key="", ffmpeg_version="")
        assert panel.auto_check_update.isChecked() is False

        panel.auto_check_update.setChecked(True)
        panel.apply(s)
        assert s.auto_check_update is True

    def test_settings_panel_show_update_info(self, qapp):
        panel = SettingsPanel()
        rel = ReleaseInfo(
            version="2.5.0",
            tag_name="v2.5.0",
            title="AutoSub Studio v2.5.0",
            changelog="Cai tien vuot troi",
            published_at="",
            asset_name="update.zip",
            asset_url="",
            asset_size=10485760,
        )

        panel.show_update_info(rel)
        assert "v2.5.0" in panel.lbl_new_version.text()
        assert "10.0 MB" in panel.lbl_update_size.text()
        assert "Cai tien vuot troi" in panel.txt_changelog.toPlainText()
        assert panel.update_details.isHidden() is False

    def test_settings_panel_confirmation_dialog_accepted(self, qapp, monkeypatch):
        panel = SettingsPanel()
        panel._available_version = "2.1.0"
        signals_emitted = []
        panel.applyUpdateRequested.connect(lambda: signals_emitted.append(True))

        monkeypatch.setattr(
            QMessageBox,
            "question",
            lambda *args, **kwargs: QMessageBox.StandardButton.Yes,
        )

        panel._on_update_now_clicked()
        assert len(signals_emitted) == 1

    def test_settings_panel_confirmation_dialog_rejected(self, qapp, monkeypatch):
        panel = SettingsPanel()
        panel._available_version = "2.1.0"
        signals_emitted = []
        panel.applyUpdateRequested.connect(lambda: signals_emitted.append(True))

        monkeypatch.setattr(
            QMessageBox,
            "question",
            lambda *args, **kwargs: QMessageBox.StandardButton.No,
        )

        panel._on_update_now_clicked()
        assert len(signals_emitted) == 0


class TestPackageReleaseScript:
    """Kiem thu script release automation: dry-run, version consistency, exclusion."""

    def test_package_release_dry_run_with_fixture(self, tmp_path):
        source_dir = tmp_path / "AutoSubStudio"
        source_dir.mkdir()
        (source_dir / "AutoSubStudio.exe").write_text("binary")
        (source_dir / "README.md").write_text("docs")

        # Thu muc Data phai bi loai bo khoi ZIP
        data_dir = source_dir / "Data"
        data_dir.mkdir()
        (data_dir / "user_project.json").write_text("user data")

        output_dir = tmp_path / "release"

        result = package_release(
            source_dir=source_dir,
            output_dir=output_dir,
            version=APP_VERSION,
            publish=False,
            dry_run=True,
        )

        zip_path = Path(str(result["zip_path"]))
        sha256_path = Path(str(result["sha256_path"]))
        manifest_path = Path(str(result["manifest_path"]))
        notes_path = Path(str(result["notes_path"]))

        assert zip_path.is_file()
        assert sha256_path.is_file()
        assert manifest_path.is_file()
        assert notes_path.is_file()

        # Kiem tra trong zip khong chua file tu thu muc Data
        with zipfile.ZipFile(zip_path, "r") as zf:
            namelist = zf.namelist()
            assert "AutoSubStudio.exe" in namelist
            assert "README.md" in namelist
            assert not any("Data" in name or "user_project" in name for name in namelist)

        # Kiem tra noi dung sha256
        sha_content = sha256_path.read_text(encoding="utf-8")
        assert str(result["sha256"]) in sha_content

        # Kiem tra manifest
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest_data["version"] == APP_VERSION
        assert manifest_data["channel"] == "stable"
        assert manifest_data["sha256"] == str(result["sha256"])
