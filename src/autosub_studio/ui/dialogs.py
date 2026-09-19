"""Cac hop thoai phu tro."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
)

from ..core.editing import Issue
from ..services import ai_gateway
from ..services.settings import Settings


class IssueDialog(QDialog):
    """Danh sach loi thoi gian, bam doi de nhay toi cau tuong ung."""

    jumpRequested = Signal(int)

    def __init__(self, issues: list[Issue], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Ket qua kiem tra lech thoi gian")
        self.resize(620, 420)
        self.list = QListWidget()
        errors = sum(1 for i in issues if i.severity == "error")
        summary = QLabel(
            f"Tim thay {len(issues)} van de ({errors} loi nang). "
            "Bam doi vao mot dong de nhay toi cau do."
            if issues
            else "Khong tim thay van de nao. Phu de dang on."
        )
        summary.setWordWrap(True)
        for issue in issues:
            item = QListWidgetItem(
                f"[{'LOI' if issue.severity == 'error' else 'Canh bao'}] "
                f"Cau {issue.index + 1}: {issue.message}"
            )
            item.setData(Qt.ItemDataRole.UserRole, issue.index)
            self.list.addItem(item)
        self.list.itemDoubleClicked.connect(self._jump)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)

        layout = QVBoxLayout(self)
        layout.addWidget(summary)
        layout.addWidget(self.list, 1)
        layout.addWidget(buttons)

    def _jump(self, item: QListWidgetItem) -> None:
        index = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(index, int):
            self.jumpRequested.emit(index)


class ShiftDialog(QDialog):
    """Nhap khoang thoi gian can doi."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Doi thoi gian phu de")
        self.edit = QLineEdit("0.5")
        self.edit.setPlaceholderText("Vi du: 0.5 hoac -1.25 (giay)")
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        row.addWidget(QLabel("Doi (giay):"))
        row.addWidget(self.edit, 1)
        layout.addLayout(row)
        layout.addWidget(QLabel("So duong = day phu de tre lai, so am = keo phu de som hon."))
        layout.addWidget(buttons)

    def value(self) -> float:
        try:
            return float(self.edit.text().replace(",", "."))
        except ValueError:
            return 0.0


class LogDialog(QDialog):
    """Cua so xem nhat ky chi tiet cua tac vu."""

    def __init__(self, title: str, text: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(760, 480)
        self.view = QPlainTextEdit()
        self.view.setReadOnly(True)
        self.view.setPlainText(text)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(self.view, 1)
        layout.addWidget(buttons)


class AIGatewayDialog(QDialog):
    """Hop thoai cau hinh ket noi AI Gateway duy nhat cua ung dung."""

    def __init__(self, settings: Settings, parent=None) -> None:
        super().__init__(parent)
        self.settings = settings
        self.setWindowTitle("Cấu hình AI Gateway")
        self.resize(580, 480)

        # 1. Endpoint & Khoa API
        conn_box = QGroupBox("Kết Nối Server AI (Chuẩn OpenAI)")
        conn_layout = QFormLayout(conn_box)
        conn_layout.setContentsMargins(10, 14, 10, 10)
        conn_layout.setSpacing(8)

        self.endpoint = QLineEdit()
        self.endpoint.setPlaceholderText("https://api.openai.com/v1 hoặc http://localhost:8000/v1")
        self.endpoint.setText(settings.ai_endpoint)
        conn_layout.addRow("Endpoint:", self.endpoint)

        self.api_key = QLineEdit()
        self.api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key.setPlaceholderText("Khóa API (lưu mã hóa bảo mật trên máy)")
        self.api_key.setText(Settings.get_secret("ai_gateway_key"))

        key_row = QHBoxLayout()
        key_row.addWidget(self.api_key, 1)
        self.show_key = QCheckBox("Hiện")
        self.show_key.toggled.connect(
            lambda checked: self.api_key.setEchoMode(
                QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password
            )
        )
        key_row.addWidget(self.show_key)
        conn_layout.addRow("Khóa API:", key_row)

        self.btn_test_conn = QPushButton("Kiểm Tra Kết Nối")
        self.btn_test_conn.clicked.connect(self._test_connection)
        conn_layout.addRow("", self.btn_test_conn)

        # 2. Cau hinh Model Sub & Prime
        model_box = QGroupBox("Cấu Hình Model & Mức Độ Suy Nghĩ (Thinking)")
        model_layout = QFormLayout(model_box)
        model_layout.setContentsMargins(10, 14, 10, 10)
        model_layout.setSpacing(8)

        # Sub
        self.model_sub = QLineEdit()
        self.model_sub.setPlaceholderText("Tên model thực tế (mặc định: sub)")
        self.model_sub.setText(settings.ai_model_sub or "sub")

        self.thinking_sub = QComboBox()
        self.thinking_sub.addItems(["none", "low", "medium", "high"])
        self.thinking_sub.setCurrentText(settings.ai_thinking_sub or "low")

        self.btn_test_sub = QPushButton("Test Model Sub")
        self.btn_test_sub.clicked.connect(self._test_sub_model)

        sub_row = QHBoxLayout()
        sub_row.addWidget(self.model_sub, 1)
        sub_row.addWidget(QLabel("Thinking:"))
        sub_row.addWidget(self.thinking_sub)
        sub_row.addWidget(self.btn_test_sub)
        model_layout.addRow("Alias 'sub':", sub_row)

        # Prime
        self.model_prime = QLineEdit()
        self.model_prime.setPlaceholderText("Tên model thực tế (mặc định: prime)")
        self.model_prime.setText(settings.ai_model_prime or "prime")

        self.thinking_prime = QComboBox()
        self.thinking_prime.addItems(["none", "low", "medium", "high"])
        self.thinking_prime.setCurrentText(settings.ai_thinking_prime or "medium")

        self.btn_test_prime = QPushButton("Test Model Prime")
        self.btn_test_prime.clicked.connect(self._test_prime_model)

        prime_row = QHBoxLayout()
        prime_row.addWidget(self.model_prime, 1)
        prime_row.addWidget(QLabel("Thinking:"))
        prime_row.addWidget(self.thinking_prime)
        prime_row.addWidget(self.btn_test_prime)
        model_layout.addRow("Alias 'prime':", prime_row)

        # 3. Trang thai ket qua kiem tra
        status_box = QGroupBox("Trạng Thái")
        status_layout = QVBoxLayout(status_box)
        self.status = QLabel("Nhập thông tin kết nối và bấm kiểm tra.")
        self.status.setWordWrap(True)
        self.status.setObjectName("Muted")
        status_layout.addWidget(self.status)

        # 4. Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_save)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(conn_box)
        layout.addWidget(model_box)
        layout.addWidget(status_box)
        layout.addWidget(buttons)

    def _set_status(self, text: str, mode: str = "Muted") -> None:
        self.status.setText(text)
        self.status.setObjectName(mode)
        self.status.style().unpolish(self.status)
        self.status.style().polish(self.status)

    def _test_connection(self) -> None:
        ep = self.endpoint.text().strip()
        key = self.api_key.text().strip()
        if not ep:
            self._set_status("Lỗi: Chưa nhập Endpoint.", "Error")
            return
        self._set_status("Đang kiểm tra kết nối...", "Muted")
        ok, msg = ai_gateway.test_connection(ep, key)
        self._set_status(msg, "Ok" if ok else "Error")

    def _test_sub_model(self) -> None:
        ep = self.endpoint.text().strip()
        key = self.api_key.text().strip()
        model = self.model_sub.text().strip() or "sub"
        thinking = self.thinking_sub.currentText()
        if not ep:
            self._set_status("Lỗi: Chưa nhập Endpoint.", "Error")
            return
        self._set_status(f"Đang kiểm tra model '{model}'...", "Muted")
        ok, msg = ai_gateway.test_model(ep, key, model=model, thinking=thinking)
        self._set_status(msg, "Ok" if ok else "Error")

    def _test_prime_model(self) -> None:
        ep = self.endpoint.text().strip()
        key = self.api_key.text().strip()
        model = self.model_prime.text().strip() or "prime"
        thinking = self.thinking_prime.currentText()
        if not ep:
            self._set_status("Lỗi: Chưa nhập Endpoint.", "Error")
            return
        self._set_status(f"Đang kiểm tra model '{model}'...", "Muted")
        ok, msg = ai_gateway.test_model(ep, key, model=model, thinking=thinking)
        self._set_status(msg, "Ok" if ok else "Error")

    def _on_save(self) -> None:
        self.settings.ai_endpoint = self.endpoint.text().strip()
        self.settings.ai_model_sub = self.model_sub.text().strip() or "sub"
        self.settings.ai_thinking_sub = self.thinking_sub.currentText()
        self.settings.ai_model_prime = self.model_prime.text().strip() or "prime"
        self.settings.ai_thinking_prime = self.thinking_prime.currentText()
        Settings.set_secret("ai_gateway_key", self.api_key.text().strip())
        self.settings.save()
        self.accept()
