"""Kiem thu cho chuc nang tu kiem tra moi truong."""

from __future__ import annotations

from autosub_studio import selftest


class TestSelfTest:
    def test_runs_and_returns_results(self):
        results, verdict = selftest.run_checks()
        assert results
        assert verdict

    def test_every_status_is_known(self):
        results, _ = selftest.run_checks()
        allowed = {selftest.OK, selftest.WARN, selftest.FAIL}
        assert {r.status for r in results} <= allowed

    def test_covers_the_important_areas(self):
        results, _ = selftest.run_checks()
        names = " | ".join(r.name for r in results)
        for needle in ("FFmpeg", "Nhan dang giong noi", "Thu muc lam viec"):
            assert needle in names

    def test_report_text_has_header_and_verdict(self):
        text = selftest.report_text()
        assert "KET QUA TU KIEM TRA" in text
        assert "KET LUAN:" in text

    def test_verdict_matches_worst_status(self):
        results, verdict = selftest.run_checks()
        if any(r.status == selftest.FAIL for r in results):
            assert "CHUA CHAY DUOC" in verdict
        elif any(r.status == selftest.WARN for r in results):
            assert verdict.startswith("CHAY DUOC")
        else:
            assert "DAY DU" in verdict

    def test_result_line_is_readable(self):
        line = selftest.CheckResult("Muc thu", selftest.OK, "chi tiet").line()
        assert "Muc thu" in line and "chi tiet" in line
