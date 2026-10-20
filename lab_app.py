"""Reaching-task lab app: session video(s) -> switches -> MediaPipe -> trials synced with the experiment
design -> reaction / movement time -> CSV + Excel.

Run:  venv/bin/python lab_app.py      (or double-click "Lab App.command" / "Lab App.bat")
Steps (left side of the window): Session -> Switches -> Process -> Blocks & parts -> Trials -> Export.
"""
import os
import sys
import time

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, QThread, Signal

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

import experiment_design as ed  # noqa: E402
import extract_kinematics as ek  # noqa: E402
import session_analysis as sa  # noqa: E402
import session_engine as se  # noqa: E402

PART_CHOICES = [ed.IGNORE] + ed.PART_ORDER
FLAG_COLOR = {'miss': '#f8d7da', 'not detected': '#e2e3e5', 'unknown': '#e2e3e5',
              'anticipation': '#fff3cd', 'corrected': '#d1ecf1'}


# -----------------------------
# Helpers
# -----------------------------
def to_pixmap(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    img = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
    return QtGui.QPixmap.fromImage(img.copy())


def read_frame(path, index=0):
    cap = cv2.VideoCapture(path)
    if index:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def mmss(ms):
    if ms is None:
        return ''
    s = ms / 1000
    return f'{int(s // 60)}:{s % 60:05.2f}'


def fmt(v, digits=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ''
    if isinstance(v, (float, np.floating)):
        return f'{0.0 if abs(v) < 0.5 * 10 ** -digits else v:.{digits}f}'
    return str(v)


class Worker(QThread):
    """Runs fn(progress, should_stop) in the background."""
    progress = Signal(float, str)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.stop_requested = False

    def run(self):
        try:
            self.done.emit(self.fn(lambda f, text='': self.progress.emit(f, text), lambda: self.stop_requested))
        except Exception as e:  # shown to the user
            import traceback
            traceback.print_exc()
            self.failed.emit(str(e))


class BoxView(QtWidgets.QGraphicsView):
    """A video frame with the numbered switch boxes; can draw one new box by dragging."""
    box_drawn = Signal(list)

    def __init__(self):
        super().__init__()
        self.setScene(QtWidgets.QGraphicsScene(self))
        self.setRenderHint(QtGui.QPainter.Antialiasing)
        self.setMinimumSize(640, 360)
        self.pix = None
        self.size_wh = (1, 1)
        self.drawing = False
        self.start = None
        self.rubber = None

    def show_frame(self, bgr, boxes):
        self.scene().clear()
        self.rubber = None
        if bgr is None:
            return
        self.size_wh = (bgr.shape[1], bgr.shape[0])
        self.pix = self.scene().addPixmap(to_pixmap(bgr))
        w, h = self.size_wh
        pen = QtGui.QPen(QtGui.QColor('#ffd400'), max(2, w // 600))
        font = QtGui.QFont()
        font.setPixelSize(max(18, h // 25))
        font.setBold(True)
        for k, (x0, y0, x1, y1) in enumerate(boxes or []):
            self.scene().addRect(x0 * w, y0 * h, (x1 - x0) * w, (y1 - y0) * h, pen)
            t = self.scene().addSimpleText(str(k + 1), font)
            t.setBrush(QtGui.QColor('#ffd400'))
            t.setPos(x0 * w, y0 * h - font.pixelSize() * 1.3)
        self.fitInView(self.scene().sceneRect(), Qt.KeepAspectRatio)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self.scene().items():
            self.fitInView(self.scene().sceneRect(), Qt.KeepAspectRatio)

    def start_drawing(self):
        self.drawing = True
        self.setCursor(Qt.CrossCursor)

    def mousePressEvent(self, e):
        if self.drawing:
            self.start = self.mapToScene(e.position().toPoint())
            self.rubber = self.scene().addRect(QtCore.QRectF(self.start, self.start),
                                               QtGui.QPen(QtGui.QColor('#00e5ff'), 4))
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self.drawing and self.start is not None and self.rubber is not None:
            self.rubber.setRect(QtCore.QRectF(self.start, self.mapToScene(e.position().toPoint())).normalized())
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if self.drawing and self.start is not None:
            r = QtCore.QRectF(self.start, self.mapToScene(e.position().toPoint())).normalized()
            self.drawing, self.start = False, None
            self.setCursor(Qt.ArrowCursor)
            w, h = self.size_wh
            if r.width() > 5 and r.height() > 5:
                self.box_drawn.emit([round(max(0, r.left()) / w, 4), round(max(0, r.top()) / h, 4),
                                     round(min(w, r.right()) / w, 4), round(min(h, r.bottom()) / h, 4)])
        super().mouseReleaseEvent(e)


def group(title, layout):
    g = QtWidgets.QGroupBox(title)
    g.setLayout(layout)
    return g


def path_row(edit, on_browse, extra=()):
    row = QtWidgets.QHBoxLayout()
    row.addWidget(edit, 1)
    b = QtWidgets.QPushButton('Browse…')
    b.clicked.connect(on_browse)
    row.addWidget(b)
    for w in extra:
        row.addWidget(w)
    return row


# -----------------------------
# Main window
# -----------------------------
class LabApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Reaching Lab - MediaPipe')
        self.resize(1400, 860)
        self.state = sa.default_state()
        self.session = None
        self.worker = None

        self.nav = QtWidgets.QListWidget()
        self.nav.setFixedWidth(190)
        self.nav.setStyleSheet('QListWidget{font-size:15px;} QListWidget::item{padding:10px;}')
        self.pages = QtWidgets.QStackedWidget()
        for name, page in [('1  Session', self.page_session()), ('2  Switches', self.page_switches()),
                           ('3  Process', self.page_process()), ('4  Blocks & parts', self.page_blocks()),
                           ('5  Trials', self.page_trials()), ('6  Export', self.page_export())]:
            self.nav.addItem(name)
            self.pages.addWidget(page)
        self.nav.currentRowChanged.connect(self.on_page)
        central = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(central)
        lay.addWidget(self.nav)
        lay.addWidget(self.pages, 1)
        self.setCentralWidget(central)
        self.status = self.statusBar()
        self.state_to_form()
        self.nav.setCurrentRow(0)

    # ---------- shared ----------
    def info(self, text):
        self.status.showMessage(text, 15000)

    def error(self, text):
        QtWidgets.QMessageBox.warning(self, 'Reaching Lab', text)

    def on_page(self, i):
        self.form_to_state()
        self.pages.setCurrentIndex(i)
        [lambda: None, self.refresh_switches, self.refresh_process, self.refresh_blocks,
         self.refresh_trials, lambda: None][i]()

    def save(self):
        self.form_to_state()
        if not self.state['subject'].strip():
            self.error('Enter a subject ID first.')
            return False
        sa.save_state(self.state)
        self.info(f'Saved to {sa.session_dir(self.state)}')
        return True

    def run_bg(self, fn, on_done, label, bar=None):
        if self.worker is not None and self.worker.isRunning():
            self.error('Something is already running - wait for it or press Cancel.')
            return
        self.worker = Worker(fn)
        bar = bar or self.proc_bar
        t0 = time.time()

        def prog(f, text):
            bar.setValue(int(f * 1000))
            el = time.time() - t0
            left = el / f - el if f > 0.01 else None
            bar.setFormat(f'{label}  {f:.0%}' + (f'  - about {left / 60:.0f} min left' if left else '') +
                          (f'  ({text})' if text else ''))
        self.worker.progress.connect(prog)
        self.worker.done.connect(lambda r: (bar.setFormat(f'{label}: done'), on_done(r)))
        self.worker.failed.connect(lambda m: (bar.setFormat(f'{label}: failed'), self.error(m)))
        bar.setRange(0, 1000)
        bar.setValue(0)
        self.worker.start()

    def cancel(self):
        if self.worker is not None:
            self.worker.stop_requested = True

    def load_session(self):
        try:
            self.session = sa.Session(self.state)
            if self.session.ready:
                self.session.analyze()
        except Exception as e:
            self.session = None
            self.error(f'Could not analyse the session: {e}')

    # ---------- 1. Session ----------
    def page_session(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        top = QtWidgets.QHBoxLayout()
        b_open = QtWidgets.QPushButton('Open a saved session…')
        b_open.clicked.connect(self.open_session)
        b_new = QtWidgets.QPushButton('New session')
        b_new.clicked.connect(self.new_session)
        b_save = QtWidgets.QPushButton('Save')
        b_save.clicked.connect(self.save)
        for b in (b_open, b_new, b_save):
            top.addWidget(b)
        top.addStretch()
        v.addLayout(top)

        f = QtWidgets.QFormLayout()
        self.e_subject = QtWidgets.QLineEdit()
        self.c_session = QtWidgets.QComboBox()
        self.c_session.addItems(['d1  (day 1: familiarization, tests, practice)', 'd2  (re-test day)'])
        f.addRow('Subject ID', self.e_subject)
        f.addRow('Session', self.c_session)
        v.addWidget(group('Participant', f))

        f = QtWidgets.QFormLayout()
        self.e_cam_a = QtWidgets.QLineEdit()
        f.addRow('Camera A video', path_row(self.e_cam_a, lambda: self.browse_video(self.e_cam_a)))
        self.e_cam_b = QtWidgets.QLineEdit()
        clr = QtWidgets.QPushButton('Clear')
        clr.clicked.connect(lambda: self.e_cam_b.clear())
        f.addRow('Camera B video (optional)', path_row(self.e_cam_b, lambda: self.browse_video(self.e_cam_b), [clr]))
        self.s_offset = QtWidgets.QDoubleSpinBox()
        self.s_offset.setRange(-3600, 3600)
        self.s_offset.setDecimals(3)
        self.s_offset.setSuffix(' s')
        b_snd = QtWidgets.QPushButton('Find from the sound')
        b_snd.clicked.connect(self.find_offset)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.s_offset)
        row.addWidget(b_snd)
        row.addWidget(QtWidgets.QLabel('seconds into video A at which video B started (0 for cameras with a shared timecode)'))
        row.addStretch()
        f.addRow('Sync offset', row)
        self.e_calib = QtWidgets.QLineEdit()
        b_click = QtWidgets.QPushButton('Click calibration…')
        b_click.clicked.connect(self.click_calibration)
        f.addRow('Camera calibration', path_row(self.e_calib, self.browse_calib, [b_click]))
        v.addWidget(group('Videos (one recording of the whole session per camera)', f))

        f = QtWidgets.QFormLayout()
        self.c_tested = QtWidgets.QComboBox()
        self.c_tested.addItems(['right', 'left'])
        self.k_contra = QtWidgets.QCheckBox('also test the contralateral (other) arm, pre and post')
        self.c_plimb = QtWidgets.QComboBox()
        self.c_plimb.addItems(['upper', 'lower'])
        self.c_pside = QtWidgets.QComboBox()
        self.c_pside.addItems(['right', 'left'])
        f.addRow('Tested arm (pre / post / re-test)', self.c_tested)
        f.addRow('', self.k_contra)
        prow = QtWidgets.QHBoxLayout()
        prow.addWidget(QtWidgets.QLabel('limb'))
        prow.addWidget(self.c_plimb)
        prow.addWidget(QtWidgets.QLabel('side'))
        prow.addWidget(self.c_pside)
        prow.addStretch()
        f.addRow('Practice (and familiarization)', prow)
        self.l_plan = QtWidgets.QLabel()
        self.l_plan.setWordWrap(True)
        f.addRow('Session plan', self.l_plan)
        for wdg in (self.c_tested, self.c_plimb, self.c_pside, self.c_session):
            wdg.currentIndexChanged.connect(self.update_plan)
        self.k_contra.toggled.connect(self.update_plan)
        v.addWidget(group('Experimental design', f))

        f = QtWidgets.QFormLayout()
        self.s_tps = QtWidgets.QSpinBox()
        self.s_tps.setRange(1, 20)
        self.s_spb = QtWidgets.QSpinBox()
        self.s_spb.setRange(1, 20)
        self.s_window = QtWidgets.QDoubleSpinBox()
        self.s_window.setRange(0.2, 5)
        self.s_window.setSuffix(' s')
        self.s_gap = QtWidgets.QDoubleSpinBox()
        self.s_gap.setRange(2, 120)
        self.s_gap.setSuffix(' s')
        self.e_pattern = QtWidgets.QLineEdit()
        self.e_pattern.setPlaceholderText('auto (the first 6 lights of the blocks), or e.g. 1,4,3,5,4,2')
        f.addRow('Trials per sequence', self.s_tps)
        f.addRow('Sequences per block', self.s_spb)
        f.addRow('Light window (miss if not hit within)', self.s_window)
        f.addRow('New block after a pause longer than', self.s_gap)
        f.addRow('Switch order in a sequence', self.e_pattern)
        adv = group('Advanced', f)
        adv.setCheckable(True)
        adv.setChecked(False)
        for i in range(f.rowCount() * 2):
            pass
        v.addWidget(adv)
        v.addStretch()
        return w

    def update_plan(self, *_):
        self.form_to_state()
        parts = sa.template_of(self.state)
        self.l_plan.setText(' → '.join(f'{p.name} ({p.blocks} blocks, {p.limb} {p.side})' for p in parts)
                            + f'   =  {sum(p.blocks for p in parts)} blocks')

    def browse_video(self, edit):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Choose a video', os.path.join(sa.PROJECT_DIR, 'media'),
                                                     'Videos (*.mov *.mp4 *.avi *.mkv *.MOV *.MP4)')
        if p:
            edit.setText(p)

    def browse_calib(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Camera calibration file',
                                                     os.path.join(sa.PROJECT_DIR, 'outputs'), 'JSON (*.json)')
        if p:
            self.e_calib.setText(p)

    def state_to_form(self):
        s = self.state
        self.e_subject.setText(s['subject'])
        self.c_session.setCurrentIndex(0 if s['session'] == 'd1' else 1)
        self.e_cam_a.setText(s['cam_a'])
        self.e_cam_b.setText(s.get('cam_b') or '')
        self.s_offset.setValue(s.get('offset_s') or 0)
        self.e_calib.setText(s.get('calibration') or '')
        d = s['design']
        self.c_tested.setCurrentText(d['tested_side'])
        self.k_contra.setChecked(d['contralateral'])
        self.c_plimb.setCurrentText(d['practice_limb'])
        self.c_pside.setCurrentText(d['practice_side'])
        st = s['settings']
        self.s_tps.setValue(st['trials_per_sequence'])
        self.s_spb.setValue(st['sequences_per_block'])
        self.s_window.setValue(st['light_window_s'])
        self.s_gap.setValue(st['block_gap_s'])
        self.e_pattern.setText(','.join(map(str, st.get('pattern') or [])))
        self.update_plan()

    def form_to_state(self):
        s = self.state
        old_a, old_b = s['cam_a'], s.get('cam_b')
        s['subject'] = self.e_subject.text().strip()
        s['session'] = 'd1' if self.c_session.currentIndex() == 0 else 'd2'
        s['cam_a'] = self.e_cam_a.text().strip()
        s['cam_b'] = self.e_cam_b.text().strip()
        if s['cam_a'] != old_a:
            s['boxes_a'] = None
        if s['cam_b'] != old_b:
            s['boxes_b'] = None
        s['offset_s'] = self.s_offset.value()
        s['calibration'] = self.e_calib.text().strip()
        s['design'] = dict(tested_side=self.c_tested.currentText(), contralateral=self.k_contra.isChecked(),
                           practice_limb=self.c_plimb.currentText(), practice_side=self.c_pside.currentText(),
                           retest=s['session'] != 'd1')
        try:
            pattern = [int(x) for x in self.e_pattern.text().replace(' ', '').split(',') if x]
        except ValueError:
            pattern = []
        s['settings'].update(trials_per_sequence=self.s_tps.value(), sequences_per_block=self.s_spb.value(),
                             light_window_s=self.s_window.value(), block_gap_s=self.s_gap.value(), pattern=pattern)

    def open_session(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, 'Session folder (results/<subject>/<session>)',
                                                            sa.RESULTS_DIR)
        if folder and os.path.exists(os.path.join(folder, 'session.json')):
            self.state = sa.load_state(folder)
            self.state_to_form()
            self.session = None
            self.info(f'Opened {folder}')
        elif folder:
            self.error('That folder has no session.json.')

    def new_session(self):
        self.state = sa.default_state(subject='', session='d1')
        self.session = None
        self.state_to_form()

    def find_offset(self):
        self.form_to_state()
        if not (self.state['cam_a'] and self.state['cam_b']):
            self.error('Choose both videos first.')
            return
        if sys.platform != 'darwin':
            self.error('Finding the offset from the sound needs macOS - type the offset in.')
            return
        import sync_videos
        try:
            off, conf = sync_videos.audio_offset(self.state['cam_a'], self.state['cam_b'])
        except Exception as e:
            self.error(f'Could not read the sound: {e}')
            return
        self.s_offset.setValue(off)
        self.info(f'Offset {off:+.3f} s (match {conf:.1f}x the next best' +
                  (' - unclear, check it' if conf < 1.5 else '') + ')')

    def click_calibration(self):
        self.form_to_state()
        if not (self.state['cam_a'] and self.state['cam_b']):
            self.error('Choose both videos (and the sync offset) first.')
            return
        if not self.save():
            return
        dlg = CalibrationDialog(self, self.state)
        if dlg.exec() and dlg.saved_path:
            self.e_calib.setText(dlg.saved_path)
            self.save()

    # ---------- 2. Switches ----------
    def page_switches(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        top = QtWidgets.QHBoxLayout()
        self.c_swcam = QtWidgets.QComboBox()
        self.c_swcam.addItems(['Camera A', 'Camera B'])
        self.c_swcam.currentIndexChanged.connect(self.refresh_switches)
        top.addWidget(self.c_swcam)
        top.addWidget(QtWidgets.QLabel('  Look for the lights from'))
        self.s_scan_from = QtWidgets.QDoubleSpinBox()
        self.s_scan_from.setRange(0, 600)
        self.s_scan_from.setSuffix(' min')
        self.s_scan_len = QtWidgets.QDoubleSpinBox()
        self.s_scan_len.setRange(0.5, 60)
        self.s_scan_len.setValue(5)
        self.s_scan_len.setSuffix(' min')
        top.addWidget(self.s_scan_from)
        top.addWidget(QtWidgets.QLabel('for'))
        top.addWidget(self.s_scan_len)
        b_find = QtWidgets.QPushButton('Find switches automatically')
        b_find.clicked.connect(self.find_switches)
        top.addWidget(b_find)
        top.addStretch()
        v.addLayout(top)
        self.l_sw = QtWidgets.QLabel('Switches are numbered 1-5 from the subject\'s RIGHT to LEFT.')
        v.addWidget(self.l_sw)
        self.boxview = BoxView()
        self.boxview.box_drawn.connect(self.on_box_drawn)
        v.addWidget(self.boxview, 1)
        bot = QtWidgets.QHBoxLayout()
        b_flip = QtWidgets.QPushButton('Flip numbering')
        b_flip.clicked.connect(self.flip_switches)
        self.s_swn = QtWidgets.QSpinBox()
        self.s_swn.setRange(1, ek.N_SWITCHES)
        b_redraw = QtWidgets.QPushButton('Redraw switch')
        b_redraw.clicked.connect(lambda: self.draw_box(self.s_swn.value() - 1))
        b_add = QtWidgets.QPushButton('Add a missing switch')
        b_add.clicked.connect(lambda: self.draw_box(None))
        b_del = QtWidgets.QPushButton('Remove switch')
        b_del.clicked.connect(self.remove_switch)
        b_cancel = QtWidgets.QPushButton('Cancel')
        b_cancel.clicked.connect(self.cancel)
        for b in (b_flip, b_redraw, self.s_swn, b_add, b_del):
            bot.addWidget(b)
        bot.addStretch()
        self.sw_bar = QtWidgets.QProgressBar()
        self.sw_bar.setTextVisible(True)
        bot.addWidget(self.sw_bar, 1)
        bot.addWidget(b_cancel)
        v.addLayout(bot)
        self.redraw_index = None
        return w

    def cur_cam(self):
        return 'b' if self.c_swcam.currentIndex() == 1 else 'a'

    def refresh_switches(self):
        cam = self.cur_cam()
        path = self.state.get(f'cam_{cam}')
        self.s_scan_from.setValue(self.state.get('scan_start_s', 0) / 60)
        self.s_scan_len.setValue(self.state.get('scan_minutes', 5))
        if not path:
            self.boxview.show_frame(None, None)
            self.l_sw.setText(f'Choose the camera {cam.upper()} video on the Session page first.')
            return
        boxes = self.state.get(f'boxes_{cam}')
        self.boxview.show_frame(read_frame(path), boxes)
        n = len(boxes or [])
        self.l_sw.setText(f'Camera {cam.upper()}: {n} of {ek.N_SWITCHES} switches. Numbered 1-5 from the '
                          f"subject's RIGHT to LEFT - use Flip if it's the other way. A box should include the light.")

    def find_switches(self):
        cam = self.cur_cam()
        path = self.state.get(f'cam_{cam}')
        if not path:
            return
        self.state['scan_start_s'] = self.s_scan_from.value() * 60
        self.state['scan_minutes'] = self.s_scan_len.value()

        def job(progress, stop):
            cap = se.open_video(path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            ok, first = cap.read()
            start = int(self.state['scan_start_s'] * fps)
            bg, pink = ek.scan_video(cap, first, total, fps, lambda f: progress(f), start_frame=start,
                                     max_frames=int(self.state['scan_minutes'] * 60 * fps), should_stop=stop)
            return ek.find_switches(bg, pink)

        def done(boxes):
            self.state[f'boxes_{cam}'] = boxes
            self.save()
            self.refresh_switches()
            if len(boxes) < ek.N_SWITCHES:
                self.error(f'Found {len(boxes)} of {ek.N_SWITCHES} switches - draw the missing one(s) with '
                           '"Add a missing switch", or look later in the video (lights must turn on while scanning).')
        self.run_bg(job, done, 'Finding switches', self.sw_bar)

    def flip_switches(self):
        cam = self.cur_cam()
        if self.state.get(f'boxes_{cam}'):
            self.state[f'boxes_{cam}'] = self.state[f'boxes_{cam}'][::-1]
            self.save()
            self.refresh_switches()

    def draw_box(self, index):
        self.redraw_index = index
        self.boxview.start_drawing()
        self.info('Drag a box around the switch (include its light).')

    def on_box_drawn(self, box):
        cam = self.cur_cam()
        boxes = list(self.state.get(f'boxes_{cam}') or [])
        if self.redraw_index is None:
            boxes = ek.sort_switches(boxes + [box])
        elif self.redraw_index < len(boxes):
            boxes[self.redraw_index] = box
        else:
            boxes.append(box)
        self.state[f'boxes_{cam}'] = boxes
        self.save()
        self.refresh_switches()

    def remove_switch(self):
        cam = self.cur_cam()
        boxes = list(self.state.get(f'boxes_{cam}') or [])
        i = self.s_swn.value() - 1
        if i < len(boxes):
            boxes.pop(i)
            self.state[f'boxes_{cam}'] = boxes
            self.save()
            self.refresh_switches()

    # ---------- 3. Process ----------
    def page_process(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.addWidget(QtWidgets.QLabel(
            'One pass over each video: MediaPipe on every frame, the switch lights, and where the hands/feet are.\n'
            'This is the slow step (on a Mac roughly 1.5 x the video\'s length); it is saved, so it runs once per video.'))
        self.l_proc = QtWidgets.QLabel()
        self.l_proc.setTextFormat(Qt.RichText)
        v.addWidget(self.l_proc)
        row = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton('Process')
        b.setStyleSheet('font-weight:bold; padding:8px 24px;')
        b.clicked.connect(self.process)
        bc = QtWidgets.QPushButton('Cancel')
        bc.clicked.connect(self.cancel)
        row.addWidget(b)
        row.addWidget(bc)
        row.addStretch()
        v.addLayout(row)
        self.proc_bar = QtWidgets.QProgressBar()
        self.proc_bar.setTextVisible(True)
        v.addWidget(self.proc_bar)
        v.addStretch()
        return w

    def refresh_process(self):
        lines = []
        for cam in ('a', 'b'):
            path = self.state.get(f'cam_{cam}')
            if not path:
                continue
            boxes = self.state.get(f'boxes_{cam}')
            c = se.load_cache(path, sa.session_dir(self.state), boxes) if boxes else None
            if not boxes:
                st = '<span style="color:#b00">switches not marked yet (step 2)</span>'
            elif c is None:
                st = 'not processed yet'
            else:
                st = (f'<span style="color:#080">processed: {c["meta"]["frames"]} frames</span>'
                      + ('' if c['meta'].get('complete') else ' <b>(stopped early - process again)</b>'))
            lines.append(f'<b>Camera {cam.upper()}</b> - {os.path.basename(path)}: {st}')
        self.l_proc.setText('<br>'.join(lines) or 'Choose the video(s) on the Session page first.')

    def process(self):
        if not self.save():
            return
        todo = []
        for cam in ('a', 'b'):
            path, boxes = self.state.get(f'cam_{cam}'), self.state.get(f'boxes_{cam}')
            if path and not boxes:
                self.error(f'Mark the switches of camera {cam.upper()} first (step 2).')
                return
            if path:
                c = se.load_cache(path, sa.session_dir(self.state), boxes)
                if c is None or not c['meta'].get('complete'):
                    todo.append((cam, path, boxes))
        if not todo:
            self.info('Already processed.')
            return
        out_dir = sa.session_dir(self.state)

        def job(progress, stop):
            for k, (cam, path, boxes) in enumerate(todo):
                se.process_video(path, boxes, out_dir,
                                 progress=lambda n, tot, left, k=k, cam=cam: progress(
                                     (k + n / max(tot, 1)) / len(todo), f'camera {cam.upper()}'),
                                 should_stop=stop)
                if stop():
                    break
            return True

        def done(_):
            self.refresh_process()
            self.session = None
            self.info('Processing finished - go to "Blocks & parts".')
        self.run_bg(job, done, 'Processing')

    # ---------- 4. Blocks ----------
    def page_blocks(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        self.l_blocks = QtWidgets.QLabel()
        self.l_blocks.setWordWrap(True)
        v.addWidget(self.l_blocks)
        self.t_blocks = QtWidgets.QTableWidget(0, 5)
        self.t_blocks.setHorizontalHeaderLabels(['Starts at', 'Lights', 'Ends at', 'Part', 'Block # in part'])
        self.t_blocks.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        v.addWidget(self.t_blocks, 1)
        row = QtWidgets.QHBoxLayout()
        b_apply = QtWidgets.QPushButton('Apply these parts')
        b_apply.clicked.connect(self.apply_parts)
        b_reset = QtWidgets.QPushButton('Back to the session plan')
        b_reset.clicked.connect(self.reset_parts)
        row.addWidget(b_apply)
        row.addWidget(b_reset)
        row.addStretch()
        v.addLayout(row)
        return w

    def ensure_session(self):
        if self.session is None or not self.session.ready or self.session.result is None:
            self.load_session()
        if self.session is None or not self.session.ready:
            return False
        return True

    def refresh_blocks(self):
        if not self.ensure_session():
            self.l_blocks.setText('Process the video(s) first (step 3).')
            self.t_blocks.setRowCount(0)
            return
        res = self.session.result
        times = self.session.A['times']
        ev = res['events']
        self.l_blocks.setText(
            f"<b>{len(res['blocks'])} groups of lights found</b>, switch order in a sequence: "
            f"<b>{', '.join(map(str, res['pattern'])) or '?'}</b>. "
            + (('<br><span style="color:#b00">' + '<br>'.join(res['warnings']) + '</span>') if res['warnings'] else
               '<br><span style="color:#080">Everything matches the session plan.</span>'))
        self.l_blocks.setTextFormat(Qt.RichText)
        self.t_blocks.setRowCount(len(res['blocks']))
        for i, (b, (part, no)) in enumerate(zip(res['blocks'], res['parts'])):
            on, off = ev[b[0]][1], ev[b[-1]][2] or ev[b[-1]][1]
            for c, text in enumerate([mmss(times[on]), str(len(b)), mmss(times[off])]):
                it = QtWidgets.QTableWidgetItem(text)
                it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                self.t_blocks.setItem(i, c, it)
            combo = QtWidgets.QComboBox()
            combo.addItems(PART_CHOICES)
            combo.setCurrentText(part)
            self.t_blocks.setCellWidget(i, 3, combo)
            spin = QtWidgets.QSpinBox()
            spin.setRange(0, 50)
            spin.setValue(no)
            self.t_blocks.setCellWidget(i, 4, spin)

    def apply_parts(self):
        n = self.t_blocks.rowCount()
        self.state['parts_override'] = [[self.t_blocks.cellWidget(i, 3).currentText(),
                                         self.t_blocks.cellWidget(i, 4).value()] for i in range(n)]
        self.save()
        self.session = None
        self.refresh_blocks()
        self.info('Parts applied.')

    def reset_parts(self):
        self.state['parts_override'] = None
        self.save()
        self.session = None
        self.refresh_blocks()

    # ---------- 5. Trials ----------
    def page_trials(self):
        w = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(w)
        left = QtWidgets.QVBoxLayout()
        flt = QtWidgets.QHBoxLayout()
        self.f_part, self.f_block, self.f_seq = QtWidgets.QComboBox(), QtWidgets.QComboBox(), QtWidgets.QComboBox()
        for lab, c in (('Part', self.f_part), ('Block', self.f_block), ('Sequence', self.f_seq)):
            flt.addWidget(QtWidgets.QLabel(lab))
            flt.addWidget(c)
            c.currentIndexChanged.connect(self.fill_trials)
        self.k_flagged = QtWidgets.QCheckBox('only marked trials')
        self.k_flagged.toggled.connect(self.fill_trials)
        flt.addWidget(self.k_flagged)
        flt.addStretch()
        left.addLayout(flt)
        cols = ['Part', 'Blk', 'Seq', 'Trial', 'Target', 'Outcome', 'Anticip.', 'Corrected', 'RT (s)', 'MT (s)', 'touse']
        self.t_trials = QtWidgets.QTableWidget(0, len(cols))
        self.t_trials.setHorizontalHeaderLabels(cols)
        self.t_trials.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.t_trials.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.t_trials.verticalHeader().setVisible(False)
        self.t_trials.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.t_trials.itemSelectionChanged.connect(self.show_trial)
        self.t_trials.itemChanged.connect(self.on_touse)
        left.addWidget(self.t_trials, 1)
        self.l_legend = QtWidgets.QLabel(
            '<span style="background:#f8d7da">&nbsp;miss&nbsp;</span> '
            '<span style="background:#fff3cd">&nbsp;started before the light&nbsp;</span> '
            '<span style="background:#d1ecf1">&nbsp;corrected (went to another switch first)&nbsp;</span> '
            '<span style="background:#e2e3e5">&nbsp;light not detected&nbsp;</span> - untick touse to leave a '
            'trial out of the summaries')
        self.l_legend.setTextFormat(Qt.RichText)
        self.l_legend.setWordWrap(True)
        left.addWidget(self.l_legend)
        lw = QtWidgets.QWidget()
        lw.setLayout(left)

        right = QtWidgets.QVBoxLayout()
        g = QtWidgets.QHBoxLayout()
        self.c_graph = QtWidgets.QComboBox()
        self.c_graph.addItems(['speed', 'x', 'y', 'z', 'path'])
        self.c_graph.currentIndexChanged.connect(self.show_trial)
        b_win = QtWidgets.QPushButton('Open graph in a new window')
        b_win.clicked.connect(self.graph_window)
        g.addWidget(QtWidgets.QLabel('Graph'))
        g.addWidget(self.c_graph)
        g.addWidget(b_win)
        g.addStretch()
        right.addLayout(g)
        self.fig = Figure(figsize=(6, 3.2), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.fig)
        right.addWidget(self.canvas, 2)
        fr = QtWidgets.QHBoxLayout()
        fr.addWidget(QtWidgets.QLabel('Video frame at:'))
        self.frame_btns = {}
        for key in ('light on', 'movement onset', 'movement end', 'trial end'):
            b = QtWidgets.QPushButton(key)
            b.clicked.connect(lambda _, k=key: self.show_video_frame(k))
            fr.addWidget(b)
        fr.addStretch()
        right.addLayout(fr)
        self.l_frame = QtWidgets.QLabel()
        self.l_frame.setAlignment(Qt.AlignCenter)
        self.l_frame.setMinimumHeight(240)
        right.addWidget(self.l_frame, 2)
        rw = QtWidgets.QWidget()
        rw.setLayout(right)
        split = QtWidgets.QSplitter()
        split.addWidget(lw)
        split.addWidget(rw)
        split.setSizes([700, 600])
        h.addWidget(split)
        self.shown_rows = []
        return w

    def refresh_trials(self):
        if not self.ensure_session():
            self.t_trials.setRowCount(0)
            return
        trials = self.session.result['trials']
        for c, key in ((self.f_part, 'Part'), (self.f_block, 'Block'), (self.f_seq, 'Sequence')):
            cur = c.currentText()
            c.blockSignals(True)
            c.clear()
            vals = list(dict.fromkeys(str(r[key]) for r in trials))
            c.addItems(['All'] + vals)
            c.setCurrentText(cur if cur in vals else 'All')
            c.blockSignals(False)
        self.fill_trials()

    def fill_trials(self, *_):
        if self.session is None or self.session.result is None:
            return
        sel = lambda c: None if c.currentText() in ('', 'All') else c.currentText()
        p, b, s = sel(self.f_part), sel(self.f_block), sel(self.f_seq)
        rows = [r for r in self.session.result['trials']
                if (p is None or r['Part'] == p) and (b is None or str(r['Block']) == b)
                and (s is None or str(r['Sequence']) == s)]
        if self.k_flagged.isChecked():
            rows = [r for r in rows if r['Outcome'] != 'hit' or r['anticipation'] or r['corrected_from']]
        self.shown_rows = rows
        t = self.t_trials
        t.blockSignals(True)
        t.setRowCount(len(rows))
        for i, r in enumerate(rows):
            vals = [r['Part'], r['Block'], r['Sequence'], r['Trial'], r['Target'], r['Outcome'],
                    f"yes ({r['anticipation_ms']:.0f} ms)" if r['anticipation'] else '',
                    f"from {r['corrected_from']}" if r['corrected_from'] else '',
                    fmt(r['responsetime']), fmt(r['movementtime'])]
            color = (FLAG_COLOR.get(r['Outcome']) or (FLAG_COLOR['anticipation'] if r['anticipation'] else None)
                     or (FLAG_COLOR['corrected'] if r['corrected_from'] else None))
            for c, val in enumerate(vals):
                it = QtWidgets.QTableWidgetItem(str(val))
                it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                if color:
                    it.setBackground(QtGui.QColor(color))
                t.setItem(i, c, it)
            it = QtWidgets.QTableWidgetItem()
            it.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            it.setCheckState(Qt.Checked if r['touse'] else Qt.Unchecked)
            t.setItem(i, len(vals), it)
        t.blockSignals(False)

    def on_touse(self, item):
        if item.column() != self.t_trials.columnCount() - 1 or item.row() >= len(self.shown_rows):
            return
        self.session.set_touse(self.shown_rows[item.row()], item.checkState() == Qt.Checked)

    def current_trial(self):
        rows = self.t_trials.selectionModel().selectedRows()
        return self.shown_rows[rows[0].row()] if rows and rows[0].row() < len(self.shown_rows) else None

    def plot_trial(self, fig, r):
        fig.clear()
        ax = fig.add_subplot(111)
        if r is None:
            return
        k, m = self.session.trial_curves(r)
        if k is None:
            ax.text(0.5, 0.5, 'no movement data for this trial', ha='center', transform=ax.transAxes)
            return
        t = k['time'] - k['stim']
        rel = lambda x: x - k['stim'] if x is not None and not np.isnan(x) else None
        kind = self.c_graph.currentText()
        on_i = end_i = None
        if m and not np.isnan(m.get('onset_s', np.nan)):
            on_i, end_i = np.searchsorted(k['time'], m['onset_s']), np.searchsorted(k['time'], m['end_s'])
        if kind == 'path':
            xyz = k['xyz']
            ax.plot(xyz[:, 0], xyz[:, 1], '.', color='0.6', ms=3)
            if on_i is not None:
                ax.plot(xyz[on_i:end_i + 1, 0], xyz[on_i:end_i + 1, 1], '-', lw=2.5, color='C0', label='movement')
            sp = self.session.switch_pos
            ax.scatter(sp[:, 0], sp[:, 1], marker='s', s=60, color='0.3')
            for i, p in enumerate(sp):
                ax.annotate(str(i + 1), p[:2], color='C3' if i + 1 == r['Target'] else '0.3', fontsize=12,
                            fontweight='bold', xytext=(6, 6), textcoords='offset points')
            if xyz.shape[1] == 2:
                ax.invert_yaxis()
            ax.set_aspect('equal', 'datalim')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
        else:
            y = k['tangvel'] if kind == 'speed' else k['xyz'][:, 'xyz'.index(kind)] if 'xyz'.index(kind) < k['xyz'].shape[1] else None
            if y is None:
                ax.text(0.5, 0.5, 'z needs two cameras', ha='center', transform=ax.transAxes)
                return
            ax.plot(t, y, '.', color='0.5', ms=3)
            if on_i is not None:
                ax.plot(t[on_i:end_i + 1], y[on_i:end_i + 1], '-', lw=2.5, color='C0')
            ax.axvline(0, color='k', ls='--', label='light on')
            if m and rel(m.get('onset_s')) is not None:
                ax.axvline(rel(m['onset_s']), color='r', ls='--', label='movement onset')
                ax.axvline(rel(m['end_s']), color='b', ls='--', label='movement end')
            end_ms = r.get('Trial_end_ms')
            if end_ms is not None:
                ax.axvline(end_ms / 1000 - k['stim'], color='g', ls='-', label='hit' if r['Outcome'] == 'hit' else 'light off')
            ax.set_xlabel('time from light on (s)')
            ax.set_ylabel('speed' if kind == 'speed' else kind)
            ax.legend(loc='upper left', fontsize=8)
        ax.set_title(f"{r['Part']}  block {r['Block']}  seq {r['Sequence']}  trial {r['Trial']}  → switch {r['Target']}"
                     f"   RT {fmt(r['responsetime'])} s  MT {fmt(r['movementtime'])} s", fontsize=9)
        if not r['touse']:
            ax.set_facecolor('0.85')

    def show_trial(self):
        r = self.current_trial()
        self.plot_trial(self.fig, r)
        self.canvas.draw_idle()
        if r is not None:
            self.show_video_frame('movement onset' if r.get('onset_s') else 'light on')

    def graph_window(self):
        r = self.current_trial()
        if r is None:
            return
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle('Trial graph')
        dlg.resize(1000, 600)
        fig = Figure(tight_layout=True)
        lay = QtWidgets.QVBoxLayout(dlg)
        lay.addWidget(FigureCanvasQTAgg(fig))
        self.plot_trial(fig, r)
        dlg.show()

    def show_video_frame(self, which):
        r = self.current_trial()
        if r is None or r.get('on_frame') is None:
            self.l_frame.clear()
            return
        times = self.session.A['times']
        if which == 'light on':
            f = r['on_frame']
        elif which == 'trial end':
            f = r['off_frame'] if r['off_frame'] is not None else r['on_frame']
        else:
            t = r.get('onset_s' if which == 'movement onset' else 'end_s')
            if t is None or np.isnan(t):
                return
            f = int(np.clip(np.searchsorted(times, t * 1000), 0, len(times) - 1))
        frame = read_frame(self.state['cam_a'], f)
        if frame is None:
            return
        # Marker and target
        j = {'upper': {'right': 16, 'left': 15}, 'lower': {'right': 28, 'left': 27}}[r['Limb']][r['Side']]
        p = self.session.A['pose'][f, j]
        sc = max(frame.shape[:2]) / 1280
        if not np.isnan(p[0]):
            cv2.drawMarker(frame, (int(p[0]), int(p[1])), (0, 0, 255), cv2.MARKER_CROSS, int(40 * sc), int(4 * sc))
        w, h = self.session.A['meta']['size']
        b = self.state['boxes_a'][r['Target'] - 1]
        M = self.session.A['motion'][f]
        c = ek.box_corners(b, w, h, 0.0) @ M[:, :2].T + M[:, 2]
        cv2.polylines(frame, [c.astype(np.int32)], True, (0, 212, 255), int(3 * sc))
        cv2.putText(frame, f'{which}  {mmss(times[f])}', (int(20 * sc), int(50 * sc)), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2 * sc, (255, 255, 255), int(3 * sc))
        pix = to_pixmap(frame)
        self.l_frame.setPixmap(pix.scaled(self.l_frame.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    # ---------- 6. Export ----------
    def page_export(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.addWidget(QtWidgets.QLabel(
            '<b>This session</b><br>'
            '&nbsp;• <i>trials.csv</i>: every trial synced with the design - part, block, sequence, trial, target, '
            'light on, touch, hit, light off, trial end, outcome (hit / miss), started before the light '
            '(anticipation), corrected (first went to another switch)<br>'
            '&nbsp;• <i>analysis.xlsx</i>: reaction time, movement time (+ AT, DT, peaks, straightness) per trial, and '
            'mean / median / SD / IQR by part, block and sequence (familiarization and touse = 0 left out)'))
        b1 = QtWidgets.QPushButton('Save this session\'s CSV + Excel')
        b1.clicked.connect(self.export_session)
        v.addWidget(b1)
        v.addSpacing(20)
        v.addWidget(QtWidgets.QLabel('<b>All subjects</b><br>One Excel file with every saved session in the '
                                     'results folder (day 1 and re-test day of a subject together).'))
        b2 = QtWidgets.QPushButton('Make the all-subjects Excel')
        b2.clicked.connect(self.export_all)
        v.addWidget(b2)
        self.l_export = QtWidgets.QLabel()
        self.l_export.setTextFormat(Qt.RichText)
        self.l_export.setOpenExternalLinks(True)
        self.l_export.setWordWrap(True)
        v.addWidget(self.l_export)
        b3 = QtWidgets.QPushButton('Open the results folder')
        b3.clicked.connect(lambda: QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(sa.RESULTS_DIR)))
        v.addWidget(b3)
        v.addStretch()
        return w

    def export_session(self):
        if not self.ensure_session():
            self.error('Process the video(s) first.')
            return
        paths = self.session.write_outputs()
        self.l_export.setText('Saved:<br>' + '<br>'.join(f'<a href="file://{p}">{p}</a>' for p in paths))

    def export_all(self):
        path, n = sa.export_all()
        self.l_export.setText(f'Saved ({n} subjects): <a href="file://{path}">{path}</a>')


# -----------------------------
# Two-camera click calibration
# -----------------------------
class ClickView(QtWidgets.QLabel):
    clicked = Signal(float, float)

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(500, 300)
        self.img = None
        self.points = []

    def set_image(self, bgr):
        self.img = bgr
        self.redraw()

    def redraw(self):
        if self.img is None:
            return
        im = self.img.copy()
        sc = max(im.shape[:2]) / 1280
        for i, (x, y) in enumerate(self.points):
            cv2.drawMarker(im, (int(x), int(y)), (0, 255, 255), cv2.MARKER_CROSS, int(30 * sc), int(3 * sc))
            cv2.putText(im, str(i + 1), (int(x + 12 * sc), int(y - 12 * sc)), cv2.FONT_HERSHEY_SIMPLEX, 1.1 * sc,
                        (0, 255, 255), int(3 * sc))
        self.setPixmap(to_pixmap(im).scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.redraw()

    def mousePressEvent(self, e):
        if self.img is None or self.pixmap() is None:
            return
        pm = self.pixmap()
        ox, oy = (self.width() - pm.width()) / 2, (self.height() - pm.height()) / 2
        x, y = (e.position().x() - ox) / pm.width() * self.img.shape[1], (e.position().y() - oy) / pm.height() * self.img.shape[0]
        if 0 <= x < self.img.shape[1] and 0 <= y < self.img.shape[0]:
            self.clicked.emit(x, y)


class CalibrationDialog(QtWidgets.QDialog):
    """Click the same points in both camera views (same moment) -> where camera B is relative to camera A."""

    def __init__(self, parent, state):
        super().__init__(parent)
        self.setWindowTitle('Two-camera calibration - click the same point in A, then in B')
        self.resize(1500, 800)
        self.state = state
        self.saved_path = None
        self.cap_a, self.cap_b = cv2.VideoCapture(state['cam_a']), cv2.VideoCapture(state['cam_b'])
        self.fps_a = self.cap_a.get(cv2.CAP_PROP_FPS) or 30
        self.first = [read_frame(state['cam_a']), read_frame(state['cam_b'])]
        v = QtWidgets.QVBoxLayout(self)
        v.addWidget(QtWidgets.QLabel(
            'Click a point in A (left), then the same point in B (right). Good points: centres of the red buttons, '
            'label corners, table corners, and the subject at this moment (knuckle, elbow) - not all on the table. '
            'At least 8 pairs, 12-20 is better.'))
        tl = QtWidgets.QHBoxLayout()
        tl.addWidget(QtWidgets.QLabel('Moment (video A)'))
        self.slider = QtWidgets.QSlider(Qt.Horizontal)
        n = int(self.cap_a.get(cv2.CAP_PROP_FRAME_COUNT))
        self.slider.setRange(0, max(0, n - 1))
        self.slider.setEnabled(True)
        self.slider.sliderReleased.connect(self.load_frames)
        tl.addWidget(self.slider, 1)
        self.l_time = QtWidgets.QLabel('0:00')
        tl.addWidget(self.l_time)
        v.addLayout(tl)
        hv = QtWidgets.QHBoxLayout()
        self.va, self.vb = ClickView(), ClickView()
        self.va.clicked.connect(lambda x, y: self.on_click('a', x, y))
        self.vb.clicked.connect(lambda x, y: self.on_click('b', x, y))
        hv.addWidget(self.va)
        hv.addWidget(self.vb)
        v.addLayout(hv, 1)
        row = QtWidgets.QHBoxLayout()
        self.l_info = QtWidgets.QLabel()
        row.addWidget(self.l_info, 1)
        row.addWidget(QtWidgets.QLabel('Lens (35 mm equiv.)'))
        self.s_focal = QtWidgets.QDoubleSpinBox()
        self.s_focal.setRange(10, 100)
        self.s_focal.setValue(28)
        row.addWidget(self.s_focal)
        for text, fn in (('Undo', self.undo), ('Compute and save', self.compute), ('Close', self.reject)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        v.addLayout(row)
        self.pts = {'a': [], 'b': []}   # first-frame coordinates
        self.motion = {'a': None, 'b': None}
        self.load_frames()

    def load_frames(self):
        fa = self.slider.value()
        t = fa / self.fps_a
        fb = int(round((t - self.state.get('offset_s', 0)) * (self.cap_b.get(cv2.CAP_PROP_FPS) or 30)))
        self.l_time.setText(f'{int(t // 60)}:{t % 60:05.2f}')
        frames = []
        for cam, cap, idx, first in (('a', self.cap_a, fa, self.first[0]), ('b', self.cap_b, max(0, fb), self.first[1])):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, fr = cap.read()
            fr = fr if ok else first
            cm = ek.CameraMotion(first)
            cm.update(fr)
            self.motion[cam] = cm.M
            frames.append(fr)
        self.va.set_image(frames[0])
        self.vb.set_image(frames[1])
        self.redraw_points()

    def to_first(self, cam, x, y):
        inv = cv2.invertAffineTransform(self.motion[cam])
        return inv[:, :2] @ np.array([x, y]) + inv[:, 2]

    def to_current(self, cam, p):
        M = self.motion[cam]
        return M[:, :2] @ np.asarray(p) + M[:, 2]

    def on_click(self, cam, x, y):
        want = 'a' if len(self.pts['a']) == len(self.pts['b']) else 'b'
        if cam != want:
            self.l_info.setText(f'Next click goes in {want.upper()}.')
            return
        self.pts[cam].append(self.to_first(cam, x, y))
        self.redraw_points()

    def undo(self):
        if len(self.pts['a']) > len(self.pts['b']):
            self.pts['a'].pop()
        elif self.pts['b']:
            self.pts['b'].pop()
        self.redraw_points()

    def redraw_points(self):
        self.va.points = [self.to_current('a', p) for p in self.pts['a']]
        self.vb.points = [self.to_current('b', p) for p in self.pts['b']]
        self.va.redraw()
        self.vb.redraw()
        n = len(self.pts['b'])
        self.l_info.setText(f'{n} pairs. Next: click in {"A" if len(self.pts["a"]) == n else "B"}. '
                            'Move the slider to use other moments too (the subject in other positions).')

    def compute(self):
        n = len(self.pts['b'])
        if n < 8:
            QtWidgets.QMessageBox.warning(self, 'Calibration', f'{n} pairs - need at least 8.')
            return
        import calibrate_cameras as cc
        pa, pb = np.array(self.pts['a'][:n]), np.array(self.pts['b'][:n])
        size_a = (self.first[0].shape[1], self.first[0].shape[0])
        size_b = (self.first[1].shape[1], self.first[1].shape[0])
        fa, fb, R, t, X, err = cc.calibrate_from_points(pa, pb, size_a, size_b, max(size_a) * self.s_focal.value() / 36,
                                                        max(size_b) * self.s_focal.value() / 36)
        out = os.path.join(sa.session_dir(self.state), 'cameras.json')
        cc.save_cameras(out, dict(K=cc.K_matrix(fa, size_a), dist=np.zeros(5), size=size_a),
                        dict(K=cc.K_matrix(fb, size_b), dist=np.zeros(5), size=size_b), R, t,
                        'relative (distance between the cameras = 1)', 'clicks',
                        dict(points=n, reprojection_px_median=round(float(np.median(err)), 2),
                             clicks_a=pa.round(1).tolist(), clicks_b=pb.round(1).tolist()))
        bad = ', '.join(f'{i + 1} ({e:.0f} px)' for i, e in enumerate(err) if e > 15)
        QtWidgets.QMessageBox.information(
            self, 'Calibration', f'Median error {np.median(err):.1f} px.' +
            (f'\nPoints that fit badly: {bad} - undo and click them again.' if bad else '\nAll points fit well.'))
        self.saved_path = out
        self.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName('Reaching Lab')
    win = LabApp()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
