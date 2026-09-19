"""Co so du lieu SQLite luu du an, lich su tac vu, kich ban va bang thuat ngu."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    event,
    inspect,
    text,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

DB_SCHEMA_VERSION = 1


class Base(DeclarativeBase):
    pass


class Project(Base):
    """Mot du an video."""

    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    folder: Mapped[str] = mapped_column(String(500), nullable=False, default="")
    video_path: Mapped[str] = mapped_column(String(500), default="")
    duration: Mapped[float] = mapped_column(Float, default=0.0)
    language: Mapped[str] = mapped_column(String(20), default="")
    target_language: Mapped[str] = mapped_column(String(20), default="vi")
    cue_count: Mapped[int] = mapped_column(Integer, default=0)
    char_count: Mapped[int] = mapped_column(Integer, default=0)
    has_subtitle: Mapped[bool] = mapped_column(Boolean, default=False)
    has_translation: Mapped[bool] = mapped_column(Boolean, default=False)
    has_dub: Mapped[bool] = mapped_column(Boolean, default=False)
    has_render: Mapped[bool] = mapped_column(Boolean, default=False)
    exported: Mapped[bool] = mapped_column(Boolean, default=False)
    current_task: Mapped[str] = mapped_column(String(120), default="")
    progress: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(200), default="Moi tao")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, onupdate=datetime.now
    )


class TaskLog(Base):
    """Lich su chay tac vu de tra cuu khi co su co."""

    __tablename__ = "task_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), default=0, index=True
    )
    name: Mapped[str] = mapped_column(String(200), default="")
    status: Mapped[str] = mapped_column(String(40), default="")
    message: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


class AutoScript(Base):
    """Kich ban chay tu dong do nguoi dung dat ten."""

    __tablename__ = "auto_scripts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(120), unique=True, nullable=False)
    steps_json: Mapped[str] = mapped_column(Text, default="[]")

    @property
    def steps(self) -> list[str]:
        try:
            data = json.loads(self.steps_json or "[]")
            return [str(s) for s in data] if isinstance(data, list) else []
        except json.JSONDecodeError:
            return []

    @steps.setter
    def steps(self, value: list[str]) -> None:
        self.steps_json = json.dumps(list(value), ensure_ascii=False)


class GlossaryTerm(Base):
    """Tu can giu nguyen hoac dich co dinh khi chuyen ngu."""

    __tablename__ = "glossary"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(200), nullable=False)
    target: Mapped[str] = mapped_column(String(200), default="")
    note: Mapped[str] = mapped_column(String(300), default="")


class TranslationCache(Base):
    """Bo nho dem ban dich de khong goi lai dich vu cho cung mot cau."""

    __tablename__ = "translation_cache"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    key: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    source_text: Mapped[str] = mapped_column(Text, default="")
    target_text: Mapped[str] = mapped_column(Text, default="")
    provider: Mapped[str] = mapped_column(String(80), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


class Meta(Base):
    """Cap khoa/gia tri dung cho phien ban luoc do va cac ghi chu he thong."""

    __tablename__ = "meta"

    key: Mapped[str] = mapped_column(String(60), primary_key=True)
    value: Mapped[str] = mapped_column(Text, default="")


@event.listens_for(Engine, "connect")
def _set_sqlite_pragma(dbapi_connection: Any, _record: Any) -> None:
    try:
        cur = dbapi_connection.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()
    except Exception:  # driver khac SQLite thi bo qua
        pass


class Database:
    """Bao boc engine va phien lam viec voi SQLite."""

    def __init__(self, db_path: str | Path) -> None:
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(
            f"sqlite:///{self.path.as_posix()}",
            future=True,
            connect_args={"check_same_thread": False, "timeout": 30},
        )
        self._session_factory = sessionmaker(bind=self.engine, expire_on_commit=False, future=True)
        Base.metadata.create_all(self.engine)
        self._migrate()

    @contextmanager
    def session(self) -> Iterator[Session]:
        """Phien lam viec tu dong commit, tu rollback khi loi."""
        s = self._session_factory()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    def _migrate(self) -> None:
        """Nang cap luoc do khi mo tep cu hon. Them cot con thieu."""
        insp = inspect(self.engine)
        with self.engine.begin() as conn:
            for table in Base.metadata.sorted_tables:
                if not insp.has_table(table.name):
                    continue
                existing = {c["name"] for c in insp.get_columns(table.name)}
                for column in table.columns:
                    if column.name in existing:
                        continue
                    ddl = (
                        f'ALTER TABLE {table.name} ADD COLUMN "{column.name}" '
                        f"{column.type.compile(self.engine.dialect)}"
                    )
                    conn.execute(text(ddl))
        with self.session() as s:
            row = s.get(Meta, "schema_version")
            if row is None:
                s.add(Meta(key="schema_version", value=str(DB_SCHEMA_VERSION)))
            else:
                row.value = str(DB_SCHEMA_VERSION)

    def ensure_default_scripts(self) -> None:
        """Tao san mot kich ban mac dinh de nguoi dung co cai de chay ngay."""
        from ..pipeline.steps import (
            DEFAULT_SCRIPT_STEPS,
            STEP_ASR,
            STEP_RENDER,
            STEP_TRANSLATE,
        )

        with self.session() as s:
            if s.query(AutoScript).count() == 0:
                script = AutoScript(name="Mac dinh")
                script.steps = list(DEFAULT_SCRIPT_STEPS)
                s.add(script)
                return
            # Nang cap dung kich ban mac dinh cu cua tool; khong sua kich ban
            # do nguoi dung tu dat hoac da tuy chinh.
            current = s.query(AutoScript).filter(AutoScript.name == "Mac dinh").one_or_none()
            if current is not None and current.steps == [STEP_ASR, STEP_TRANSLATE, STEP_RENDER]:
                current.steps = list(DEFAULT_SCRIPT_STEPS)

    def recover_interrupted_projects(self) -> int:
        """Xoa trang thai dang chay gia con sot lai sau khi app bi tat."""
        recovered = 0
        with self.session() as s:
            rows = s.query(Project).filter(Project.current_task != "").all()
            for row in rows:
                row.current_task = ""
                row.progress = 0
                row.status = "Đã dừng khi tool đóng trước đó"
                recovered += 1
        return recovered
