"""Bang mau va bieu kieu giao dien."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication

from ..services.paths import fonts_dir

# Nen
BG = "#30353a"
BG_DEEP = "#202428"
PANEL = "#30353a"
PANEL_2 = "#252a2e"
BORDER = "#52616a"
BORDER_SOFT = "#455159"

# Chu
TEXT = "#e8e8e8"
MUTED = "#a0a0a0"
VALUE = "#ffd54a"  # gia tri trong o chon
LABEL = "#dcdcdc"

# Nhan manh
GREEN = "#079b70"
GREEN_HOVER = "#0eae7e"
GREEN_DARK = "#087c5c"
BLUE = "#168ac0"
BLUE_HOVER = "#209bd0"
YELLOW = "#f1dd16"
RED = "#e05252"
OLIVE = "#7d6b35"
ORANGE = "#e8952f"

# Trang thai trong bang du an
OK_TEXT = "#3ecf8e"
BAD_TEXT = "#e86464"
WARN_TEXT = "#ffcc55"
GREY_TEXT = "#9a9a9a"

_active_font_family: str | None = None
_active_font_path: Path | None = None
_font_registered: bool = False


def _reset_font_state() -> None:
    """Xoa trang thai font da dang ky (dung cho kiem thu)."""
    global _active_font_family, _active_font_path, _font_registered, QSS
    _active_font_family = None
    _active_font_path = None
    _font_registered = False
    QSS = build_qss()


def find_ui_font_file() -> Path | None:
    """Tim tep font Unicode kem theo ung dung."""
    fdir = fonts_dir()
    if fdir is None or not fdir.is_dir():
        return None
    regular = fdir / "NotoSans-Regular.ttf"
    if regular.is_file():
        return regular
    for cand in sorted(fdir.glob("NotoSans*.ttf")):
        if cand.is_file():
            return cand
    for ext in ("*.ttf", "*.otf"):
        for cand in sorted(fdir.glob(ext)):
            if cand.is_file():
                return cand
    return None


def get_fallback_font_family(app: QApplication | None = None) -> str:
    """Lay font mac dinh cua he thong thay vi gia tri co dinh khi nap font that bai."""
    try:
        sys_font = QFontDatabase.systemFont(QFontDatabase.SystemFont.GeneralFont)
        fam = sys_font.family()
        if fam:
            return fam
    except Exception:
        pass
    target = app or QApplication.instance()
    if target is not None and isinstance(target, QApplication):
        fam = target.font().family()
        if fam:
            return fam
    return "Sans Serif"


def get_ui_font_family() -> str:
    """Ten ho font dang duoc su dung cho giao dien."""
    global _active_font_family
    if _active_font_family:
        return _active_font_family
    return get_fallback_font_family()


def get_ui_font_path() -> Path | None:
    """Duong dan tep font da dang ky thanh cong, hoac None neu dung fallback."""
    return _active_font_path


def init_app_font(
    app: QApplication | None = None,
    *,
    font_path: Path | str | None = None,
    force_fallback: bool = False,
) -> str:
    """Dang ky font Unicode (Noto Sans) va thiet lap font mac dinh cho ung dung."""
    global _active_font_family, _active_font_path, _font_registered, QSS

    target_app = app if isinstance(app, QApplication) else QApplication.instance()
    q_app = target_app if isinstance(target_app, QApplication) else None
    if _font_registered and _active_font_family and not font_path and not force_fallback:
        if q_app is not None:
            q_app.setFont(QFont(_active_font_family, 9))
        return _active_font_family

    family: str | None = None
    selected_path: Path | None = None

    if not force_fallback:
        if font_path is not None:
            p = Path(font_path)
            selected_path = p if p.is_file() else None
        else:
            selected_path = find_ui_font_file()

        if selected_path and selected_path.is_file():
            try:
                font_id = QFontDatabase.addApplicationFont(str(selected_path.resolve()))
                if font_id >= 0:
                    families = QFontDatabase.applicationFontFamilies(font_id)
                    if families:
                        family = families[0]
                        fdir = selected_path.parent
                        for extra_name in (
                            "NotoSans-Bold.ttf",
                            "NotoSans-SemiBold.ttf",
                            "NotoSans-Medium.ttf",
                            "NotoSans-Italic.ttf",
                            "NotoSans-BoldItalic.ttf",
                        ):
                            extra_file = fdir / extra_name
                            if extra_file.is_file() and extra_file != selected_path:
                                QFontDatabase.addApplicationFont(str(extra_file.resolve()))
            except Exception:
                family = None

    if not family:
        family = get_fallback_font_family(q_app)
        selected_path = None

    _active_font_family = family
    _active_font_path = selected_path
    _font_registered = True

    if q_app is not None:
        q_app.setFont(QFont(family, 9))

    QSS = build_qss(family)
    return family


def build_qss(font_family: str | None = None) -> str:
    """Tao ma CSS giao dien dua tren ho font duoc chon."""
    fam = font_family or get_ui_font_family()
    return f"""
* {{ font-family: "{fam}", sans-serif; font-size: 12px; }}
QWidget {{ background: {BG}; color: {TEXT}; }}
QMainWindow, QDialog {{ background: {BG}; }}

/* ------------------------------------------------------------ khung nhom */
QGroupBox {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 0px;
    margin-top: 11px;
    padding: 12px 8px 8px 8px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: {MUTED};
    font-weight: 600;
}}
QFrame#Card {{
    background: {PANEL};
    border: 1px solid {BORDER_SOFT};
    border-radius: 4px;
}}
QFrame#Sunken {{
    background: {BG_DEEP};
    border: 1px solid {BORDER};
    border-radius: 3px;
}}

/* ------------------------------------------------------------ nhan */
QLabel {{ background: transparent; color: {LABEL}; font-weight: 600; }}
QLabel#Muted {{ color: {MUTED}; font-weight: normal; }}
QLabel#Value {{ color: {VALUE}; font-weight: 600; }}
QLabel#Ok {{ color: {OK_TEXT}; font-weight: 600; }}
QLabel#Bad {{ color: {BAD_TEXT}; font-weight: 600; }}
QLabel#Empty {{ color: {MUTED}; font-weight: normal; font-size: 13px; }}
QLabel#SectionTitle {{ color: {MUTED}; font-weight: 600; }}

/* ------------------------------------------------------------ nut */
QPushButton {{
    background: {GREEN};
    border: none;
    border-radius: 11px;
    padding: 4px 14px;
    color: #ffffff;
    font-weight: 600;
    min-height: 18px;
}}
QPushButton:hover {{ background: {GREEN_HOVER}; }}
QPushButton:pressed {{ background: {GREEN_DARK}; }}
QPushButton:disabled {{ background: #3f3f3f; color: #7a7a7a; }}

QPushButton#Blue {{ background: {BLUE}; }}
QPushButton#Blue:hover {{ background: {BLUE_HOVER}; }}
QPushButton#Danger {{ background: #b3453f; }}
QPushButton#Danger:hover {{ background: #c8524b; }}
QPushButton#Flat {{
    background: transparent; border: 1px solid {BORDER};
    border-radius: 3px; color: {TEXT}; padding: 4px 10px;
}}
QPushButton#Flat:hover {{ background: {PANEL_2}; }}
QPushButton#Icon {{
    background: transparent; border: none; border-radius: 3px;
    padding: 2px; min-height: 0;
}}
QPushButton#Icon:hover {{ background: {PANEL_2}; }}
QFrame#RenderToolbar, QFrame#RenderTools {{
    background: {PANEL}; border: 1px solid {BORDER}; border-radius: 0px;
}}
QPushButton#RenderIcon {{
    background: transparent; border: none; border-radius: 2px;
    padding: 1px; color: #48d9c0; font-size: 14px; font-weight: 800;
}}
QPushButton#RenderIcon:hover {{ background: {PANEL_2}; color: {YELLOW}; }}

/* ------------------------------------------------------------ o nhap */
QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox {{
    background: {BG_DEEP};
    border: 1px solid {BORDER};
    border-radius: 0px;
    padding: 3px 6px;
    color: {TEXT};
    selection-background-color: {GREEN};
    selection-color: #ffffff;
}}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus,
QSpinBox:focus, QDoubleSpinBox:focus {{ border-color: {GREEN}; }}
QLineEdit:disabled, QSpinBox:disabled {{ color: #7a7a7a; }}

QComboBox {{
    background: {BG_DEEP};
    border: 1px solid {BORDER};
    border-radius: 0px;
    padding: 3px 6px;
    color: {VALUE};
    font-weight: 600;
}}
QComboBox:focus {{ border-color: {GREEN}; }}
QComboBox::drop-down {{
    border: none; background: transparent; width: 16px;
    subcontrol-origin: padding; subcontrol-position: center right;
}}
QComboBox::down-arrow {{
    width: 0; height: 0; background: transparent;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {MUTED};
    margin-right: 4px;
}}
QComboBox::down-arrow:on {{ border-top-color: {YELLOW}; }}
QComboBox QAbstractItemView {{
    background: {PANEL};
    border: 1px solid {BORDER};
    color: {TEXT};
    selection-background-color: {GREEN};
    selection-color: #ffffff;
    outline: none;
}}

QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    background: {PANEL_2}; border: none; width: 13px;
}}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
    width: 0; height: 0; background: transparent;
    border-left: 3px solid transparent; border-right: 3px solid transparent;
    border-bottom: 4px solid {TEXT};
}}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    width: 0; height: 0; background: transparent;
    border-left: 3px solid transparent; border-right: 3px solid transparent;
    border-top: 4px solid {TEXT};
}}

/* ------------------------------------------------------------ bang */
QTableView, QTreeView, QListView {{
    background: {BG};
    border: 1px solid {BORDER};
    gridline-color: {BORDER_SOFT};
    selection-background-color: #2f5c46;
    selection-color: {TEXT};
    alternate-background-color: #343a3f;
    outline: none;
}}
QHeaderView::section {{
    background: {BG_DEEP};
    color: {TEXT};
    border: none;
    border-right: 1px solid {BORDER};
    border-bottom: 1px solid {BORDER};
    padding: 5px 6px;
    font-weight: 700;
}}
QTableView::item {{ padding: 2px 5px; }}
QTableCornerButton::section {{ background: {PANEL_2}; border: none; }}

/* Hai bang chinh dung font, do dam va mau chon sat voi giao dien mau NTS. */
QTableView#CueTable, QTableView#ProjectTable {{
    font-family: "{fam}", sans-serif;
    font-size: 12px;
    font-weight: 600;
}}
QTableView#CueTable {{
    selection-background-color: #075742;
    selection-color: #ffeb00;
}}
QTableView#ProjectTable {{
    color: #ffffff;
    selection-background-color: #0c503e;
    selection-color: #ffe400;
}}
QTableView#CueTable QHeaderView::section,
QTableView#ProjectTable QHeaderView::section {{
    font-family: "{fam}", sans-serif;
    font-size: 12px;
    font-weight: 700;
    color: #eeeeee;
}}

/* ------------------------------------------------------------ tab hang giua */
QTabWidget::pane {{ border: none; top: 0; }}
QTabBar {{ alignment: center; qproperty-drawBase: 0; }}
QTabBar::tab {{
    background: {BLUE};
    color: #ffffff;
    border: none;
    border-radius: 12px;
    padding: 6px 14px;
    margin: 2px 3px;
    font-weight: 700;
}}
QTabBar::tab:hover {{ background: {BLUE_HOVER}; }}
QTabBar::tab:selected {{
    background: {PANEL};
    color: {YELLOW};
    border-bottom: 2px solid {YELLOW};
}}

/* ------------------------------------------------------------ khac */
QProgressBar {{
    background: {BG_DEEP}; border: 1px solid {BORDER}; border-radius: 3px;
    text-align: center; height: 14px; color: {TEXT};
}}
QProgressBar::chunk {{ background: {GREEN}; }}

QSlider::groove:horizontal {{ background: {BORDER}; height: 4px; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {GREEN}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    background: {YELLOW}; width: 12px; height: 12px;
    margin: -5px 0; border-radius: 6px;
}}

QCheckBox, QRadioButton {{ background: transparent; color: {LABEL}; font-weight: 600; }}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 12px; height: 12px;
    border: 1px solid #f0c000; border-radius: 2px; background: {BG_DEEP};
}}
QCheckBox::indicator:checked {{ background: {ORANGE}; border: 2px solid {BG_DEEP}; }}
QRadioButton::indicator {{ border-radius: 7px; }}
QRadioButton::indicator:checked {{ background: {GREEN}; border-color: {GREEN}; }}

QScrollBar:vertical {{ background: {BG}; width: 11px; margin: 0; }}
QScrollBar::handle:vertical {{ background: #575757; border-radius: 5px; min-height: 24px; }}
QScrollBar::handle:vertical:hover {{ background: #6a6a6a; }}
QScrollBar:horizontal {{ background: {BG}; height: 11px; margin: 0; }}
QScrollBar::handle:horizontal {{ background: #575757; border-radius: 5px; min-width: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QSplitter::handle {{ background: {BG}; }}
QSplitter::handle:horizontal {{ width: 5px; }}
QSplitter::handle:vertical {{ height: 5px; }}

QStatusBar {{ background: {PANEL}; border-top: 1px solid {BORDER_SOFT}; color: {MUTED}; }}
QStatusBar::item {{ border: none; }}
QToolTip {{
    background: {PANEL}; color: {TEXT};
    border: 1px solid {BORDER}; padding: 4px; border-radius: 3px;
}}
QMenu {{ background: {PANEL}; border: 1px solid {BORDER}; padding: 3px; }}
QMenu::item {{ padding: 5px 20px; border-radius: 3px; }}
QMenu::item:selected {{ background: {GREEN}; color: #ffffff; }}
"""


def get_qss(font_family: str | None = None) -> str:
    """Lay ma QSS hien tai hoac theo font chi dinh."""
    if font_family is not None:
        return build_qss(font_family)
    return QSS


QSS = build_qss()
