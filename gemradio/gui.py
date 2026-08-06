"""The receiver: a retro front panel for GemRadio.

Warm walnut and amber, a lit dial, a pair of VU needles, an ON AIR lamp that
comes up when the DJ opens the mic, and the cover of whatever is playing.
The whole panel is repainted from a station snapshot on a timer, so no Qt
object is ever touched from an audio or model thread.
"""

from __future__ import annotations

import atexit
import math
import signal
import sys
from pathlib import Path

from PyQt5.QtCore import (QPointF, QRectF, Qt, QTimer, pyqtSignal)
from PyQt5.QtGui import (QBrush, QColor, QFont, QFontDatabase, QIcon,
                         QLinearGradient, QPainter, QPainterPath, QPen, QPixmap,
                         QRadialGradient)
from PyQt5.QtWidgets import (QApplication, QFrame, QHBoxLayout, QLabel,
                             QPushButton, QSizePolicy, QSlider, QVBoxLayout,
                             QWidget)

from . import config
from .library import Library
from .logging_util import get_logger
from .station import Station

log = get_logger(__name__)

# -- palette ---------------------------------------------------------------
WALNUT_DARK = QColor(28, 20, 15)
WALNUT = QColor(46, 32, 23)
PANEL = QColor(38, 29, 22)
CREAM = QColor(232, 214, 178)
AMBER = QColor(255, 176, 58)
AMBER_DIM = QColor(150, 96, 30)
GLASS = QColor(24, 18, 13)
RED_ON_AIR = QColor(228, 70, 48)
GREEN_LAMP = QColor(120, 220, 140)

COVER_SIZE = 232


def _font(size: int, weight: int = QFont.Normal, mono: bool = False,
          spacing: float = 0.0) -> QFont:
    families = (["DejaVu Sans Mono", "Liberation Mono", "Courier New"] if mono
                else ["Optima", "Gill Sans", "DejaVu Sans", "Liberation Sans"])
    available = set(QFontDatabase().families())
    family = next((f for f in families if f in available), families[-1])
    f = QFont(family, size, weight)
    if spacing:
        f.setLetterSpacing(QFont.PercentageSpacing, 100 + spacing)
    return f


# --------------------------------------------------------------------------
# Widgets
# --------------------------------------------------------------------------

class DialWidget(QWidget):
    """Backlit tuning dial; the needle drifts with the music level."""

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(96)
        self.level = 0.0
        self._needle = 0.5
        self.station_text = f"{config.STATION_NAME}  {config.STATION_FREQUENCY}"
        self.lit = False

    def set_level(self, level: float, lit: bool) -> None:
        self.level = level
        self.lit = lit
        target = 0.5 + (level - 0.25) * 0.30
        self._needle += (max(0.08, min(0.92, target)) - self._needle) * 0.12
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(2, 2, -2, -2)

        glass = QLinearGradient(r.topLeft(), r.bottomLeft())
        base = QColor(46, 36, 24) if self.lit else QColor(26, 21, 16)
        glass.setColorAt(0.0, base.lighter(125))
        glass.setColorAt(0.5, base)
        glass.setColorAt(1.0, base.darker(140))
        path = QPainterPath()
        path.addRoundedRect(r, 8, 8)
        p.fillPath(path, QBrush(glass))

        if self.lit:
            glow = QRadialGradient(QPointF(r.center().x(), r.bottom()), r.width() * 0.55)
            glow.setColorAt(0.0, QColor(255, 170, 60, 62))
            glow.setColorAt(1.0, QColor(255, 170, 60, 0))
            p.fillPath(path, QBrush(glow))

        p.setPen(QPen(QColor(90, 66, 40), 1.5))
        p.drawPath(path)

        # frequency scale
        ink = AMBER if self.lit else AMBER_DIM
        p.setPen(QPen(ink, 1))
        p.setFont(_font(7, mono=True))
        span = r.width() - 36
        for i in range(41):
            x = r.left() + 18 + span * i / 40
            major = i % 5 == 0
            h = 11 if major else 5
            p.drawLine(QPointF(x, r.top() + 12), QPointF(x, r.top() + 12 + h))
            if major:
                p.drawText(QRectF(x - 16, r.top() + 24, 32, 12), Qt.AlignCenter,
                           f"{88 + i // 5 * 3}")

        p.setFont(_font(12, QFont.Bold, spacing=18))
        p.setPen(QPen(CREAM if self.lit else QColor(120, 104, 80)))
        p.drawText(QRectF(r.left(), r.bottom() - 30, r.width(), 22),
                   Qt.AlignCenter, self.station_text)

        x = r.left() + 18 + span * self._needle
        p.setPen(QPen(QColor(255, 90, 60, 220), 2))
        p.drawLine(QPointF(x, r.top() + 8), QPointF(x, r.bottom() - 32))
        p.setPen(QPen(QColor(255, 150, 120, 90), 6))
        p.drawLine(QPointF(x, r.top() + 8), QPointF(x, r.bottom() - 32))


class VUMeter(QWidget):
    """Ballistic needle meter, one per channel."""

    def __init__(self, label: str):
        super().__init__()
        self.label = label
        self.value = 0.0
        self._shown = 0.0
        self._peak = 0.0
        self.setMinimumSize(120, 78)

    def set_value(self, v: float) -> None:
        self.value = max(0.0, min(1.0, v))
        # Slow rise, slower fall: classic VU ballistics.
        self._shown += (self.value - self._shown) * (0.35 if self.value > self._shown else 0.12)
        self._peak = max(self._peak * 0.985, self._shown)
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        path = QPainterPath()
        path.addRoundedRect(r, 5, 5)
        grad = QLinearGradient(r.topLeft(), r.bottomLeft())
        grad.setColorAt(0.0, QColor(238, 222, 186))
        grad.setColorAt(1.0, QColor(206, 184, 142))
        p.fillPath(path, QBrush(grad))
        p.setPen(QPen(QColor(70, 50, 32), 1.2))
        p.drawPath(path)

        pivot = QPointF(r.center().x(), r.bottom() - 6)
        radius = min(r.width() * 0.45, r.height() * 0.86)
        start, end = math.radians(208), math.radians(-28)

        for i in range(11):
            t = i / 10
            ang = start + (end - start) * t
            hot = t > 0.72
            p.setPen(QPen(QColor(178, 46, 32) if hot else QColor(74, 54, 36),
                          2.0 if i % 5 == 0 else 1.0))
            outer = QPointF(pivot.x() + math.cos(ang) * radius,
                            pivot.y() - math.sin(ang) * radius)
            inner = QPointF(pivot.x() + math.cos(ang) * (radius - (8 if i % 5 == 0 else 5)),
                            pivot.y() - math.sin(ang) * (radius - (8 if i % 5 == 0 else 5)))
            p.drawLine(inner, outer)

        p.setFont(_font(7, QFont.Bold, spacing=25))
        p.setPen(QPen(QColor(96, 70, 44)))
        p.drawText(QRectF(r.left(), r.bottom() - 20, r.width(), 12), Qt.AlignCenter,
                   f"VU  {self.label}")

        ang = start + (end - start) * self._shown
        tip = QPointF(pivot.x() + math.cos(ang) * (radius - 3),
                      pivot.y() - math.sin(ang) * (radius - 3))
        p.setPen(QPen(QColor(30, 22, 16), 1.8))
        p.drawLine(pivot, tip)
        p.setBrush(QBrush(QColor(40, 30, 20)))
        p.setPen(Qt.NoPen)
        p.drawEllipse(pivot, 3.2, 3.2)

        if self._peak > 0.02:
            pang = start + (end - start) * self._peak
            ptip = QPointF(pivot.x() + math.cos(pang) * (radius - 3),
                           pivot.y() - math.sin(pang) * (radius - 3))
            p.setPen(QPen(QColor(190, 60, 40, 150), 1.4))
            p.drawLine(QPointF(pivot.x() + math.cos(pang) * (radius - 12),
                               pivot.y() - math.sin(pang) * (radius - 12)), ptip)


class Lamp(QWidget):
    """Small indicator lamp with a caption."""

    def __init__(self, text: str, color: QColor, width: int = 78):
        super().__init__()
        self.text = text
        self.color = color
        self.on = False
        self.setFixedSize(width, 26)

    def set_on(self, on: bool) -> None:
        if on != self.on:
            self.on = on
            self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        path = QPainterPath()
        path.addRoundedRect(r, 4, 4)
        col = self.color if self.on else QColor(58, 46, 36)
        p.fillPath(path, QBrush(col.darker(260) if not self.on else col.darker(150)))
        if self.on:
            glow = QRadialGradient(r.center(), r.width() * 0.7)
            glow.setColorAt(0.0, QColor(col.red(), col.green(), col.blue(), 120))
            glow.setColorAt(1.0, QColor(col.red(), col.green(), col.blue(), 0))
            p.fillPath(path, QBrush(glow))
        p.setPen(QPen(QColor(96, 74, 50), 1))
        p.drawPath(path)
        p.setFont(_font(7, QFont.Bold, spacing=30))
        p.setPen(QPen(QColor(255, 236, 200) if self.on else QColor(112, 94, 74)))
        p.drawText(r, Qt.AlignCenter, self.text)


class CoverArt(QLabel):
    def __init__(self):
        super().__init__()
        self.setFixedSize(COVER_SIZE, COVER_SIZE)
        self.setAlignment(Qt.AlignCenter)
        self._path = ""
        self._placeholder()

    def _placeholder(self) -> None:
        """A record with no sleeve: concentric grooves on dark card."""
        size = COVER_SIZE
        mid = size / 2
        pm = QPixmap(size, size)
        pm.fill(QColor(30, 23, 18))
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(QColor(96, 72, 46), 1.5))
        p.drawRoundedRect(QRectF(2, 2, size - 4, size - 4), 6, 6)
        p.setPen(QPen(QColor(120, 92, 58)))
        p.setBrush(Qt.NoBrush)
        for rr in (0.31, 0.22, 0.12):
            p.drawEllipse(QPointF(mid, mid), size * rr, size * rr)
        p.setBrush(QBrush(QColor(150, 112, 60)))
        p.drawEllipse(QPointF(mid, mid), 6, 6)
        p.end()
        self.setPixmap(pm)

    def show_cover(self, path: str) -> None:
        if path == self._path:
            return
        self._path = path
        if not path or not Path(path).is_file():
            self._placeholder()
            return
        pm = QPixmap(path)
        if pm.isNull():
            self._placeholder()
            return
        size = COVER_SIZE
        scaled = pm.scaled(size, size, Qt.KeepAspectRatioByExpanding,
                           Qt.SmoothTransformation)
        if scaled.width() != size or scaled.height() != size:
            x = max(0, (scaled.width() - size) // 2)
            y = max(0, (scaled.height() - size) // 2)
            scaled = scaled.copy(x, y, size, size)
        self.setPixmap(scaled)


class ProgressBar(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedHeight(6)
        self.value = 0.0

    def set_value(self, v: float) -> None:
        v = max(0.0, min(1.0, v))
        if abs(v - self.value) > 0.0005:
            self.value = v
            self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect())
        p.fillRect(r, QColor(26, 20, 15))
        if self.value > 0:
            grad = QLinearGradient(r.topLeft(), r.topRight())
            grad.setColorAt(0.0, AMBER_DIM)
            grad.setColorAt(1.0, AMBER)
            p.fillRect(QRectF(r.left(), r.top(), r.width() * self.value, r.height()),
                       QBrush(grad))


class SectionLabel(QLabel):
    def __init__(self, text: str):
        super().__init__(text)
        self.setFont(_font(7, QFont.Bold, spacing=40))
        self.setStyleSheet("color: #8a6f4c;")


# --------------------------------------------------------------------------
# Main window
# --------------------------------------------------------------------------

class RadioWindow(QWidget):
    def __init__(self, station: Station):
        super().__init__()
        self.station = station
        self.setWindowTitle(f"{config.STATION_NAME} — offline radio")
        self.setMinimumSize(960, 720)
        self._marquee_offset = 0
        self._build()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._refresh)
        self.timer.start(60)

    # -- layout -------------------------------------------------------------
    def _build(self) -> None:
        self.setStyleSheet(f"""
            QWidget {{ background: {WALNUT_DARK.name()}; color: {CREAM.name()}; }}
            QLabel  {{ background: transparent; }}
            QPushButton {{
                background: #3a2b20; border: 1px solid #6b5030; border-radius: 6px;
                color: #f0dcb4; padding: 9px 20px; font-weight: bold;
                letter-spacing: 2px;
            }}
            QPushButton:hover {{ background: #4a3728; }}
            QPushButton:pressed {{ background: #2c2018; }}
            QPushButton:disabled {{ color: #6d5b44; border-color: #4a3a2a; }}
            QPushButton:checked {{
                background: #5a4118; border-color: #b98a3a; color: #ffdca0;
            }}
            QSlider::groove:horizontal {{
                height: 4px; background: #241b14; border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                background: #d9a24e; width: 13px; margin: -6px 0; border-radius: 6px;
            }}
            QSlider::sub-page:horizontal {{ background: #8a5f2a; border-radius: 2px; }}
        """)

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(12)

        # -- top: brand + lamps
        top = QHBoxLayout()
        brand = QVBoxLayout()
        brand.setSpacing(0)
        name = QLabel(config.STATION_NAME)
        name.setFont(_font(26, QFont.Bold, spacing=60))
        name.setStyleSheet(f"color: {CREAM.name()};")
        sub = QLabel("O F F L I N E   B R O A D C A S T")
        sub.setFont(_font(7, QFont.Bold, spacing=55))
        sub.setStyleSheet("color: #8a6f4c;")
        brand.addWidget(name)
        brand.addWidget(sub)
        top.addLayout(brand)
        top.addStretch(1)

        self.lamp_air = Lamp("ON AIR", RED_ON_AIR, 86)
        self.lamp_en = Lamp("EN", GREEN_LAMP, 44)
        self.lamp_fr = Lamp("FR", GREEN_LAMP, 44)
        self.lamp_it = Lamp("IT", GREEN_LAMP, 44)
        for w in (self.lamp_air, self.lamp_en, self.lamp_fr, self.lamp_it):
            top.addWidget(w)
        root.addLayout(top)

        self.dial = DialWidget()
        root.addWidget(self.dial)

        # -- middle: cover + now playing
        mid = QHBoxLayout()
        mid.setSpacing(18)

        left = QVBoxLayout()
        left.setSpacing(8)
        self.cover = CoverArt()
        left.addWidget(self.cover, 0, Qt.AlignTop)
        left.addSpacing(4)
        left.addWidget(SectionLabel("PLAYED EARLIER"))
        self.recent_label = QLabel("")
        self.recent_label.setFont(_font(8))
        self.recent_label.setStyleSheet("color: #8d7452;")
        self.recent_label.setWordWrap(True)
        self.recent_label.setFixedWidth(COVER_SIZE)
        self.recent_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        left.addWidget(self.recent_label)
        left.addStretch(1)
        mid.addLayout(left, 0)

        info = QVBoxLayout()
        info.setSpacing(3)
        info.addWidget(SectionLabel("PROGRAMME"))
        self.show_label = QLabel("—")
        self.show_label.setFont(_font(15, QFont.Bold))
        self.show_label.setStyleSheet(f"color: {AMBER.name()};")
        info.addWidget(self.show_label)
        self.tagline_label = QLabel("")
        self.tagline_label.setFont(_font(9))
        self.tagline_label.setStyleSheet("color: #a58a63;")
        info.addWidget(self.tagline_label)
        info.addSpacing(10)

        info.addWidget(SectionLabel("NOW PLAYING"))
        self.title_label = QLabel("—")
        self.title_label.setFont(_font(17, QFont.Bold))
        self.title_label.setWordWrap(True)
        info.addWidget(self.title_label)
        self.artist_label = QLabel("")
        self.artist_label.setFont(_font(12))
        self.artist_label.setStyleSheet("color: #cbb182;")
        self.artist_label.setWordWrap(True)
        info.addWidget(self.artist_label)
        self.album_label = QLabel("")
        self.album_label.setFont(_font(9))
        self.album_label.setStyleSheet("color: #977d59;")
        self.album_label.setWordWrap(True)
        info.addWidget(self.album_label)

        info.addSpacing(8)
        self.progress = ProgressBar()
        info.addWidget(self.progress)
        self.time_label = QLabel("0:00 / 0:00")
        self.time_label.setFont(_font(8, mono=True))
        self.time_label.setStyleSheet("color: #8a6f4c;")
        info.addWidget(self.time_label)
        info.addStretch(1)
        mid.addLayout(info, 1)

        meters = QVBoxLayout()
        meters.setSpacing(8)
        self.vu_l = VUMeter("L")
        self.vu_r = VUMeter("R")
        meters.addWidget(self.vu_l)
        meters.addWidget(self.vu_r)
        meters.addStretch(1)
        mid.addLayout(meters, 0)
        root.addLayout(mid, 0)

        # -- what the DJ is saying, and what was said before
        dj_box = QFrame()
        dj_box.setStyleSheet(
            "background: #221a13; border: 1px solid #4b3927; border-radius: 6px;"
        )
        dj_layout = QVBoxLayout(dj_box)
        dj_layout.setContentsMargins(12, 10, 12, 10)
        dj_layout.setSpacing(8)

        mic = SectionLabel("MICROPHONE")
        mic.setStyleSheet("color: #7d6444; border: none;")
        dj_layout.addWidget(mic)

        self.dj_label = QLabel("")
        self.dj_label.setFont(_font(12))
        self.dj_label.setWordWrap(True)
        self.dj_label.setMinimumHeight(46)
        self.dj_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.dj_label.setStyleSheet("color: #f2c987; border: none;")
        dj_layout.addWidget(self.dj_label)

        self.dj_log = QLabel("")
        self.dj_log.setFont(_font(8))
        self.dj_log.setWordWrap(True)
        self.dj_log.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.dj_log.setStyleSheet("color: #8a7150; border: none;")
        dj_layout.addWidget(self.dj_log, 1)
        root.addWidget(dj_box, 1)

        # -- ticker
        self.ticker = QLabel("")
        self.ticker.setFont(_font(8, mono=True))
        self.ticker.setStyleSheet("color: #8a6f4c;")
        root.addWidget(self.ticker)

        # -- controls
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color: #4a3728;")
        root.addWidget(line)

        controls = QHBoxLayout()
        controls.setSpacing(10)
        self.btn_play = QPushButton("▶  PLAY")
        self.btn_play.setMinimumWidth(140)
        self.btn_play.clicked.connect(self._toggle)
        controls.addWidget(self.btn_play)

        self.btn_skip = QPushButton("▶▶  NEXT")
        self.btn_skip.clicked.connect(self.station.skip)
        self.btn_skip.setEnabled(False)
        controls.addWidget(self.btn_skip)

        self.btn_power = QPushButton("◐  LOW POWER")
        self.btn_power.setCheckable(True)
        self.btn_power.setChecked(self.station.profile.key == "low")
        self.btn_power.setToolTip(
            "Smallest installed model, longer runs of music between links, "
            "one voice at a time, no Whisper pass."
        )
        self.btn_power.clicked.connect(self._toggle_power)
        controls.addWidget(self.btn_power)

        controls.addSpacing(16)
        vol_label = QLabel("VOLUME")
        vol_label.setFont(_font(7, QFont.Bold, spacing=40))
        vol_label.setStyleSheet("color: #8a6f4c;")
        controls.addWidget(vol_label)
        self.volume = QSlider(Qt.Horizontal)
        self.volume.setRange(0, 100)
        self.volume.setValue(int(config.MASTER_VOLUME * 100))
        self.volume.setFixedWidth(150)
        self.volume.valueChanged.connect(lambda v: self.station.set_volume(v / 100.0))
        controls.addWidget(self.volume)

        controls.addStretch(1)
        self.status_label = QLabel("Ready")
        self.status_label.setFont(_font(8, mono=True))
        self.status_label.setStyleSheet("color: #8a6f4c;")
        self.status_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.status_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        controls.addWidget(self.status_label)
        root.addLayout(controls)

    # -- painting the wooden case ------------------------------------------
    def paintEvent(self, _):
        p = QPainter(self)
        r = QRectF(self.rect())
        grad = QLinearGradient(r.topLeft(), r.bottomRight())
        grad.setColorAt(0.0, WALNUT)
        grad.setColorAt(0.45, PANEL)
        grad.setColorAt(1.0, WALNUT_DARK)
        p.fillRect(r, QBrush(grad))
        p.setPen(QPen(QColor(70, 52, 34, 90), 1))
        for y in range(0, int(r.height()), 4):     # faint wood grain
            p.drawLine(0, y, int(r.width()), y)

    # -- refresh ------------------------------------------------------------
    def _toggle_power(self) -> None:
        low = self.btn_power.isChecked()
        self.station.set_profile("low" if low else "full")
        self.btn_power.setText("◐  LOW POWER" if low else "◑  FULL POWER")

    def _toggle(self) -> None:
        if self.station.running:
            self.station.stop()
            self.btn_play.setText("▶  PLAY")
        else:
            self.btn_play.setEnabled(False)
            self.btn_play.setText("…")
            QApplication.processEvents()
            self.station.start()
            self.btn_play.setText("■  STOP")
            self.btn_play.setEnabled(True)

    @staticmethod
    def _mmss(seconds: float) -> str:
        seconds = max(0, int(seconds))
        return f"{seconds // 60}:{seconds % 60:02d}"

    def _refresh(self) -> None:
        s = self.station.snapshot()
        running = s["running"]

        self.btn_play.setText("■  STOP" if running else "▶  PLAY")
        self.btn_skip.setEnabled(running and bool(s["next_ready"]))

        vu_l, vu_r = s["vu"]
        self.vu_l.set_value(vu_l)
        self.vu_r.set_value(vu_r)
        self.dial.set_level((vu_l + vu_r) / 2, running)

        self.lamp_air.set_on(bool(s["speaking"]))
        lang = s["language"]
        self.lamp_en.set_on(running and lang == "en")
        self.lamp_fr.set_on(running and lang == "fr")
        self.lamp_it.set_on(running and lang == "it")

        self.show_label.setText(s["show"] or "—")
        self.tagline_label.setText(s["tagline"])
        self.title_label.setText(s["title"] or ("—" if running else "Press play"))
        artist = s["artist"]
        if s["year"]:
            artist = f"{artist}  ·  {s['year']}" if artist else s["year"]
        self.artist_label.setText(artist)
        album = s["album"]
        if s["genre"]:
            album = f"{album}  ·  {s['genre']}" if album else s["genre"]
        self.album_label.setText(album)
        self.cover.show_cover(s["cover"])
        self.recent_label.setText("\n".join(f"·  {t}" for t in s.get("recent", [])))

        duration = s["duration"] or 0.0
        position = s["position"] or 0.0
        self.progress.set_value(position / duration if duration > 0 else 0.0)
        self.time_label.setText(f"{self._mmss(position)} / {self._mmss(duration)}")

        log_lines = s["speech_log"]
        if s["on_air_text"]:
            voice = s["on_air_voice"]
            self.dj_label.setText(f"🎙  {voice}:  {s['on_air_text']}" if voice
                                  else f"🎙  {s['on_air_text']}")
            history = log_lines[:-1]
        elif log_lines:
            when, text = log_lines[-1]
            self.dj_label.setText(f"{when}   {text}")
            history = log_lines[:-1]
        else:
            self.dj_label.setText("—" if running else "")
            history = []
        self.dj_log.setText("\n".join(f"{w}   {t}" for w, t in history[-3:][::-1]))

        bits = []
        if s["next_title"]:
            bits.append(("NEXT ▸ " if s["next_ready"] else "PREPARING ▸ ") + s["next_title"])
        if s["language_label"]:
            bits.append(s["language_label"])
        if s["crossfading"]:
            bits.append("CROSSFADE")
        self.ticker.setText("     ·     ".join(bits))

        status = [s["status"], s["profile_label"]]
        if s["scanning"]:
            done, total = s["scan_progress"]
            status.append(f"indexing {done}/{total}")
        status.append(f"DJ: {s['brain']}")
        self.status_label.setText("   |   ".join(status))

    def closeEvent(self, event):
        self.timer.stop()
        self.setEnabled(False)
        self.status_label.setText("Shutting down…")
        QApplication.processEvents()
        self.station.shutdown()
        event.accept()


def run(profile: str | None = None) -> int:
    app = QApplication(sys.argv)
    app.setApplicationName(config.STATION_NAME)
    app.setDesktopFileName("gemradio")
    icon = Path(__file__).with_name("assets") / "gemradio.svg"
    if icon.is_file():
        app.setWindowIcon(QIcon(str(icon)))

    library = Library()
    station = Station(library, profile=profile)

    # Whatever ends the process — window close, Ctrl-C, SIGTERM from a session
    # logout — the station has to put everything down behind it.
    atexit.register(station.shutdown)

    def _signal_quit(signum, _frame):
        log.info("received signal %s", signum)
        app.quit()

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(sig, _signal_quit)
        except (ValueError, OSError, AttributeError):
            pass
    # Qt's event loop blocks Python signal delivery unless it wakes up now and
    # then, so give the interpreter a regular slot to run its handlers.
    heartbeat = QTimer()
    heartbeat.timeout.connect(lambda: None)
    heartbeat.start(300)

    app.aboutToQuit.connect(station.shutdown)

    window = RadioWindow(station)
    window.resize(1020, 760)
    if icon.is_file():
        window.setWindowIcon(QIcon(str(icon)))
    window.show()

    if library.count() < 50:
        library.scan_async()
    else:
        # Refresh the index quietly in the background on every launch.
        library.scan_async()

    if not config.WHISPER_ENABLED:
        log.info("whisper listening disabled")

    station.start()
    return app.exec_()
