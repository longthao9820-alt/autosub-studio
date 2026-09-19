"""Bang mau va bieu kieu giao dien."""

from __future__ import annotations

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

QSS = f"""
* {{ font-family: "Segoe UI", "Tahoma", sans-serif; font-size: 12px; }}
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
    font-family: "Tahoma";
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
    font-family: "Tahoma";
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
