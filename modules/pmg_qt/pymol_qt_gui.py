"""
Contains main class for PyMOL QT GUI
"""


from collections import defaultdict
import os
import re
import sys

import pymol
import pymol._gui
from pymol import colorprinting, save_shortcut

from pymol.Qt import QtGui, QtCore, QtWidgets
from pymol.Qt.utils import (getSaveFileNameWithExt, UpdateLock, WidgetMenu,
        MainThreadCaller,
        PopupOnException,
        connectFontContextMenu, getMonospaceFont)

from .pymol_gl_widget import PyMOLGLWidget
from . import keymapping

from pmg_qt import properties_dialog, file_dialogs

Qt = QtCore.Qt
QFileDialog = QtWidgets.QFileDialog
getOpenFileNames = QFileDialog.getOpenFileNames


class PyMOLQtGUI(QtWidgets.QMainWindow, pymol._gui.PyMOLDesktopGUI):
    '''
    PyMOL QMainWindow GUI
    '''

    from pmg_qt.file_dialogs import (
            load_dialog,
            load_mae_dialog,
            file_fetch_pdb,
            file_save_png,
            file_save_mpeg,
            file_save_map,
            file_save_aln,
            file_save
    )

    _ext_window_visible = True
    _initialdir = ''

    def keyPressEvent(self, ev):
        if self.handle_casp_rank_key(ev):
            return

        args = keymapping.keyPressEventToPyMOLButtonArgs(ev)

        if args is not None:
            self.pymolwidget.pymol.button(*args)

    def closeEvent(self, event):
        self.cmd.quit()

    # for thread-safe viewport command
    viewportsignal = QtCore.Signal(int, int)

    def pymolviewport(self, w, h):
        cw, ch = self.cmd.get_viewport()
        pw = self.pymolwidget
        scale = pw.fb_scale

        # maintain aspect ratio
        if h < 1:
            if w < 1:
                pw.pymol.reshape(int(scale * pw.width()),
                                 int(scale * pw.height()), True)
                return
            h = (w * ch) / cw
        if w < 1:
            w = (h * cw) / ch

        win_size = self.size()
        delta = QtCore.QSize(w - cw, h - ch) / scale

        # window resize
        self.resize(delta + win_size)

    def get_view(self):
        self.cmd.get_view(2, quiet=0)
        QtWidgets.QApplication.clipboard().setText(self.cmd.get_view(3))
        print(" get_view: matrix copied to clipboard.")

    def __init__(self):  # noqa
        QtWidgets.QMainWindow.__init__(self)
        self.setDockOptions(QtWidgets.QMainWindow.DockOption.AllowTabbedDocks |
                            QtWidgets.QMainWindow.DockOption.AllowNestedDocks)

        # resize Window before it is shown
        options = pymol.invocation.options
        self.resize(
            options.win_x + (220 if options.internal_gui else 0),
            options.win_y + (246 if options.external_gui else 18))

        # for thread-safe viewport command
        self.viewportsignal.connect(self.pymolviewport)

        # reusable dialogs
        self.dialog_png = None
        self.advanced_settings_dialog = None
        self.props_panel = None
        self.builder = None
        self.shortcut_menu_filter_dialog = None
        self.scene_panel_dialog = None
        self.ai_panel = None
        self.casp_toolbar = None
        self._casp_restore_state = None
        self._casp_pending_model_name = None
        self.casp_rank_labels = []
        self.casp_rank_clear_buttons = []

        # setting index -> callable
        self.setting_callbacks = defaultdict(list)

        # "session_file" setting in window title
        self.setting_callbacks[440].append(
            lambda v: self.setWindowTitle("PyMOL (" + os.path.basename(v) + ")")
        )

        # "External" Command Line and Loggin Widget
        self._setup_history()
        self.lineedit = CommandLineEdit()
        self.lineedit.setObjectName("command_line")
        self.browser = QtWidgets.QPlainTextEdit()
        self.browser.setObjectName("feedback_browser")
        self.browser.setReadOnly(True)

        # convenience: clicking into feedback browser gives focus to command
        # line. Drawback: Copying with CTRL+C doesn't work in feedback
        # browser -> clear focus proxy while text selected
        self.browser.setFocusProxy(self.lineedit)

        @self.browser.copyAvailable.connect
        def _(yes):
            self.browser.setFocusProxy(None if yes else self.lineedit)
            self.browser.setFocus()

        # Font
        self.browser.setFont(getMonospaceFont())
        connectFontContextMenu(self.browser)

        lineeditlayout = QtWidgets.QHBoxLayout()
        command_label = QtWidgets.QLabel("PyMOL>")
        command_label.setObjectName("command_label")
        lineeditlayout.addWidget(command_label)
        lineeditlayout.addWidget(self.lineedit)
        self.lineedit.setToolTip('''Command Input Area

Get the list of commands by hitting <TAB>

Get the list of arguments for one command with a question mark:
PyMOL> color ?

Read the online help for a command with "help":
PyMOL> help color

Get autocompletion for many arguments by hitting <TAB>
PyMOL> color ye<TAB>    (will autocomplete "yellow")
''')

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.browser)
        layout.addLayout(lineeditlayout)

        quickbuttonslayout = QtWidgets.QVBoxLayout()
        quickbuttonslayout.setSpacing(2)

        extguilayout = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.Direction.LeftToRight)
        extguilayout.setContentsMargins(2, 2, 2, 2)
        extguilayout.addLayout(layout)
        extguilayout.addLayout(quickbuttonslayout)

        class ExtGuiFrame(QtWidgets.QFrame):
            def mouseDoubleClickEvent(_, event):
                self.toggle_ext_window_dockable(True)

            _size_hint = QtCore.QSize(options.win_x, options.ext_y)

            def sizeHint(self):
                return self._size_hint

        dockWidgetContents = ExtGuiFrame(self)
        dockWidgetContents.setLayout(extguilayout)
        dockWidgetContents.setObjectName("extgui")

        self.ext_window = \
            dockWidget = QtWidgets.QDockWidget(self)
        dockWidget.setWindowTitle("External GUI")
        dockWidget.setWidget(dockWidgetContents)
        if options.external_gui:
            dockWidget.setTitleBarWidget(QtWidgets.QWidget())
        else:
            dockWidget.hide()

        self.addDockWidget(Qt.DockWidgetArea.TopDockWidgetArea, dockWidget)

        # rearrange vertically if docking left or right
        @dockWidget.dockLocationChanged.connect
        def _(area):
            if area == Qt.DockWidgetArea.LeftDockWidgetArea or area == Qt.DockWidgetArea.RightDockWidgetArea:
                extguilayout.setDirection(QtWidgets.QBoxLayout.Direction.BottomToTop)
                quickbuttonslayout.takeAt(quickbuttons_stretch_index)
            else:
                extguilayout.setDirection(QtWidgets.QBoxLayout.Direction.LeftToRight)
                if quickbuttons_stretch_index >= quickbuttonslayout.count():
                    quickbuttonslayout.addStretch()

        # OpenGL Widget
        self.pymolwidget = PyMOLGLWidget(self)
        self.setCentralWidget(self.pymolwidget)

        cmd = self.cmd = self.pymolwidget.cmd

        '''
        # command completion
        completer = QtWidgets.QCompleter(cmd.kwhash.keywords, self)
        self.lineedit.setCompleter(completer)
        '''

        # overload <Tab> action
        self.lineedit.installEventFilter(self)
        self.pymolwidget.installEventFilter(self)

        # Quick Buttons
        for row in [
            [
                ('Reset', cmd.reset),
                ('Zoom', lambda: cmd.zoom(animate=1.0)),
                ('Orient', lambda: cmd.orient(animate=1.0)),

                # render dialog will be constructed when the menu is shown
                # for the first time. This way it's populated with the current
                # viewport and settings. Also defers parsing of the ui file.
                ('Draw/Ray', WidgetMenu(self).setSetupUi(self.render_dialog)),
            ],
            [
                ('Unpick', cmd.unpick),
                ('Deselect', cmd.deselect),
                ('Rock', cmd.rock),
                ('Get View', self.get_view),
            ],
            [
                ('|<', cmd.rewind),
                ('<', cmd.backward),
                ('Stop', cmd.mstop),
                ('Play', cmd.mplay),
                ('>', cmd.forward),
                ('>|', cmd.ending),
                ('MClear', cmd.mclear),
            ],
            [
                ('Builder', self.open_builder_panel),
                ('Properties', self.open_props_dialog),
                ('Rebuild', cmd.rebuild),
            ],
        ]:
            hbox = QtWidgets.QHBoxLayout()
            hbox.setSpacing(2)

            for name, callback in row:
                btn = QtWidgets.QPushButton(name)
                btn.setProperty("quickbutton", True)
                btn.setAttribute(Qt.WidgetAttribute.WA_LayoutUsesWidgetRect) # OS X workaround
                hbox.addWidget(btn)

                if callback is None:
                    btn.setEnabled(False)
                elif isinstance(callback, QtWidgets.QMenu):
                    btn.setMenu(callback)
                else:
                    btn.released.connect(callback)

            quickbuttonslayout.addLayout(hbox)

        self._setup_casp_toolbar(cmd)

        # progress bar
        hbox = QtWidgets.QHBoxLayout()
        self.progressbar = QtWidgets.QProgressBar()
        self.progressbar.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Minimum,
                QtWidgets.QSizePolicy.Policy.Minimum)
        hbox.addWidget(self.progressbar)
        self.abortbutton = QtWidgets.QPushButton('Abort')
        self.abortbutton.setStyleSheet("background: #FF0000; color: #FFFFFF")
        self.abortbutton.released.connect(cmd.interrupt)
        hbox.addWidget(self.abortbutton)
        quickbuttonslayout.addLayout(hbox)

        quickbuttonslayout.addStretch()
        quickbuttons_stretch_index = quickbuttonslayout.count() - 1

        # menu top level
        self.menubar = menubar = self.menuBar()

        # action groups
        actiongroups = {}

        def _addmenu(data, menu):
            '''Fill a menu from "data"'''
            menu.setTearOffEnabled(True)
            menu.setWindowTitle(menu.title())  # needed for Windows
            for item in data:
                if item[0] == 'separator':
                    menu.addSeparator()
                elif item[0] == 'menu':
                    _addmenu(item[2], menu.addMenu(item[1].replace('&', '&&')))
                elif item[0] == 'command':
                    command = item[2]
                    if command is None:
                        print('warning: skipping', item)
                    else:
                        if isinstance(command, str):
                            command = lambda c=command: cmd.do(c)
                        menu.addAction(item[1], command)
                elif item[0] == 'check':
                    if len(item) > 4:
                        menu.addAction(
                            SettingAction(self, cmd, item[2], item[1],
                                          item[3], item[4]))
                    else:
                        menu.addAction(
                            SettingAction(self, cmd, item[2], item[1]))
                elif item[0] == 'radio':
                    label, name, value = item[1:4]
                    try:
                        group, type_, values = actiongroups[item[2]]
                    except KeyError:
                        group = QtWidgets.QActionGroup(self)
                        type_, values = cmd.get_setting_tuple(name)
                        actiongroups[item[2]] = group, type_, values
                    action = QtWidgets.QAction(label, self)
                    action.triggered.connect(lambda _=0, args=(name, value):
                                             cmd.set(*args, log=1, quiet=0))

                    self.setting_callbacks[cmd.setting._get_index(
                        name)].append(
                            lambda v, V=value, a=action: a.setChecked(v == V))

                    group.addAction(action)
                    menu.addAction(action)
                    action.setCheckable(True)
                    if values[0] == value:
                        action.setChecked(True)
                elif item[0] == 'open_recent_menu':
                    self.open_recent_menu = menu.addMenu('Open Recent...')
                else:
                    print('error:', item)

        # recent files menu
        self.open_recent_menu = None

        # for plugins
        self.menudict = {'': menubar}

        # menu
        for _, label, data in self.get_menudata(cmd):
            assert _ == 'menu'
            menu = menubar.addMenu(label)
            self.menudict[label] = menu
            _addmenu(data, menu)

        # hack for macOS to hide "Edit > Start Dictation"
        # https://bugreports.qt.io/browse/QTBUG-43217
        if pymol.IS_MACOS:
            self.menudict['Edit'].setTitle('Edit_')
            QtCore.QTimer.singleShot(10, lambda:
                    self.menudict['Edit'].setTitle('Edit'))

        # recent files menu
        if self.open_recent_menu:
            @self.open_recent_menu.aboutToShow.connect
            def _():
                self.open_recent_menu.clear()
                for fname in self.recent_filenames:
                    self.open_recent_menu.addAction(
                            fname if len(fname) < 128 else '...' + fname[-120:],
                            lambda fname=fname: self.load_dialog(fname))

        # some experimental window control
        menu = self.menudict['Display'].addSeparator()
        menu = self.menudict['Display'].addMenu('External GUI')
        menu.addAction('Toggle dockable', self.toggle_ext_window_dockable).setShortcut(
            QtGui.QKeySequence('Ctrl+E'))

        ext_vis_action = self.ext_window.toggleViewAction()
        ext_vis_action.setText('Visible')
        menu.addAction(ext_vis_action)

        # extra key mappings (MacPyMOL compatible)
        QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+O'), self).activated.connect(self.file_open)
        QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+S'), self).activated.connect(self.session_save)

        # feedback
        self.feedback_timer = QtCore.QTimer()
        self.feedback_timer.setSingleShot(True)
        self.feedback_timer.timeout.connect(self.update_feedback)
        self.feedback_timer.start(100)

        # legacy plugin system
        self.menudict['Plugin'].addAction(
            'Initialize Plugin System', self.initializePlugins)

        # focus in command line
        if options.external_gui:
            self.lineedit.setFocus()
        else:
            self.pymolwidget.setFocus()

        # Apply PyMOL stylesheet
        try:
            with open(cmd.exp_path('$PYMOL_DATA/pmg_qt/styles/pymol.sty')) as f:
                style = f.read()
        except IOError:
            print('Could not read PyMOL stylesheet.')
            print('DEBUG: PYMOL_DATA=' + repr(os.getenv('PYMOL_DATA')))
            style = ""

        if style:
            self.setStyleSheet(style)

        # Load saved shortcuts on launch
        self.saved_shortcuts = pymol.save_shortcut.load_and_set(self.cmd)

    def lineeditKeyPressEventFilter(self, watched, event):
        key = event.key()
        if key == Qt.Key.Key_Tab:
            self.complete()
        elif key == Qt.Key.Key_Up:
            if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                self.back_search()
            else:
                self.back()
        elif key == Qt.Key.Key_Down:
            self.forward()
        elif key == Qt.Key.Key_Return or key == Qt.Key.Key_Enter:
            # filter out "Return" instead of binding lineedit.returnPressed,
            # because otherwise OrthoKey would capture it as well.
            self.doPrompt()
        else:
            return False
        return True

    def eventFilter(self, watched, event):
        '''
        Filter out <Tab> event to do tab-completion instead of move focus
        '''
        type_ = event.type()
        if type_ == QtCore.QEvent.Type.KeyRelease:
            if event.key() == Qt.Key.Key_Tab:
                # silently skip tab release
                return True
        elif type_ == QtCore.QEvent.Type.KeyPress:
            if watched is self.lineedit:
                return self.lineeditKeyPressEventFilter(watched, event)
            elif event.key() == Qt.Key.Key_Tab:
                self.keyPressEvent(event)
                return True
        return False

    def toggle_ext_window_dockable(self, neverfloat=False):
        '''
        Toggle whether the "external" GUI is dockable
        '''
        dockWidget = self.ext_window

        if dockWidget.titleBarWidget() is None:
            tbw = QtWidgets.QWidget()
        else:
            tbw = None

        dockWidget.setFloating(tbw is None and not neverfloat)
        dockWidget.setTitleBarWidget(tbw)
        dockWidget.show()

    def toggle_fullscreen(self, toggle=-1):
        '''
        Full screen
        '''
        is_fullscreen = self.windowState() == Qt.WindowState.WindowFullScreen

        if toggle == -1:
            toggle = not is_fullscreen

        if not is_fullscreen:
            self._ext_window_visible = self.ext_window.isVisible()

        if toggle:
            self.menubar.hide()
            if not self.ext_window.isFloating():
                self.ext_window.hide()
            self.showFullScreen()
            self.pymolwidget.setFocus()
        else:
            self.menubar.show()
            if self._ext_window_visible:
                self.ext_window.show()
            self.showNormal()

    @property
    def initialdir(self):
        '''
        Be in sync with cd/pwd on the console until the first file has been
        browsed, then remember the last directory.
        '''
        return self._initialdir or os.getcwd()

    @initialdir.setter
    def initialdir(self, value):
        self._initialdir = value

    ##################
    # UI Forms
    ##################

    def load_form(self, name, dialog=None):
        '''Load a form from pmg_qt/forms/{name}.py'''
        import importlib
        if dialog is None:
            dialog = QtWidgets.QDialog(self)
            widget = dialog
        elif dialog == 'floating':
            widget = QtWidgets.QWidget(self)
        else:
            widget = dialog

        try:
            m = importlib.import_module('.forms.' + name, 'pmg_qt')
        except ImportError as e:
            if pymol.Qt.DEBUG:
                print('load_form import failed (%s)' % (e,))
            uifile = os.path.join(os.path.dirname(__file__), 'forms', '%s.ui' % name)
            form = pymol.Qt.utils.loadUi(uifile, widget)
        else:
            if hasattr(m, 'Ui_Form'):
                form = m.Ui_Form()
            else:
                form = m.Ui_Dialog()

            form.setupUi(widget)

        if dialog == 'floating':
            dialog = QtWidgets.QDockWidget(widget.windowTitle(), self)
            dialog.setFloating(True)
            dialog.setWidget(widget)
            dialog.resize(widget.size())

        form._dialog = dialog
        return form

    def edit_colors_dialog(self):
        form = self.load_form('colors')
        form.list_colors.setSortingEnabled(True)

        # populate list with named colors
        for color_index in self.cmd.get_color_indices():
            form.list_colors.addItem(color_index[0])

        # update spinboxes for given color
        def load_color(name):
            index = self.cmd.get_color_index(name)
            if index == -1:
                return
            rgb = self.cmd.get_color_tuple(index)
            form.input_R.setValue(rgb[0])
            form.input_G.setValue(rgb[1])
            form.input_B.setValue(rgb[2])

        # update spinbox from slider
        spinbox_lock = [False]
        def update_spinbox(spinbox, value):
            if not spinbox_lock[0]:
                spinbox.setValue(value / 100.)

        # update sliders and colored frame
        def update_gui(*args):
            spinbox_lock[0] = True
            R = form.input_R.value()
            G = form.input_G.value()
            B = form.input_B.value()
            form.slider_R.setValue(round(R * 100))
            form.slider_G.setValue(round(G * 100))
            form.slider_B.setValue(round(B * 100))
            form.frame_color.setStyleSheet(
                "background-color: rgb(%d,%d,%d)" % (
                    R * 0xFF, G * 0xFF, B * 0xFF))
            spinbox_lock[0] = False

        def run():
            name  = form.input_name.text()
            R = form.input_R.value()
            G = form.input_G.value()
            B = form.input_B.value()

            self.cmd.do('set_color %s, [%.2f, %.2f, %.2f]\nrecolor' %
                        (name, R, G, B))

            # if new color, insert and make current row
            if not form.list_colors.findItems(name, Qt.MatchFlag.MatchExactly):
                form.list_colors.addItem(name)
                form.list_colors.setCurrentItem(
                    form.list_colors.findItems(name, Qt.MatchFlag.MatchExactly)[0])

        # hook up events
        form.slider_R.valueChanged.connect(lambda v: update_spinbox(form.input_R, v))
        form.slider_G.valueChanged.connect(lambda v: update_spinbox(form.input_G, v))
        form.slider_B.valueChanged.connect(lambda v: update_spinbox(form.input_B, v))
        form.input_R.valueChanged.connect(update_gui)
        form.input_G.valueChanged.connect(update_gui)
        form.input_B.valueChanged.connect(update_gui)
        form.input_name.textChanged.connect(load_color)
        form.list_colors.currentTextChanged.connect(form.input_name.setText)
        form.button_apply.clicked.connect(run)

        form._dialog.show()

    def open_builder_panel(self):
        from pmg_qt.builder import BuilderPanelDocked
        from pymol import plugins

        app = plugins.get_pmgapp()
        if not self.builder:
            self.builder = BuilderPanelDocked(self, app)
            self.addDockWidget(Qt.DockWidgetArea.TopDockWidgetArea, self.builder)

        self.builder.show()
        self.builder.raise_()

    def open_props_dialog(self):
        from .properties_dialog import PropsDialog

        if not self.props_panel:
            self.props_panel = PropsDialog(self)

        self.props_panel.get_dialog().show()
        self.props_panel.get_dialog().raise_()

    def edit_pymolrc(self):
        from . import TextEditor
        from pymol import plugins
        TextEditor.edit_pymolrc(plugins.get_pmgapp())

    ##################
    # Menu callbacks
    ##################

    def file_open(self):
        fnames = getOpenFileNames(self, 'Open file', self.initialdir)[0]
        partial = 0
        for fname in fnames:
            if not self.load_dialog(fname, partial=partial):
                break
            partial = 1

    def session_save(self):
        fname = self.cmd.get('session_file')
        fname = self.cmd.as_pathstr(fname)
        return self.session_save_as(fname)

    @PopupOnException.decorator
    def session_save_as(self, fname=''):
        formats = [
            'PyMOL Session File (*.pse *.pze *.pse.gz)',
            'PyMOL Show File (*.psw *.pzw *.psw.gz)',
        ]
        if not fname:
            fname = getSaveFileNameWithExt(
                self,
                'Save Session As...',
                self.initialdir,
                filter=';;'.join(formats))
        if fname:
            self.initialdir = os.path.dirname(fname)
            self.cmd.save(fname, format='pse', quiet=0)
            self.recent_filenames_add(fname)

    def render_dialog(self, widget=None):
        form = self.load_form('render', widget)
        lock = UpdateLock([ZeroDivisionError])

        def get_factor():
            units = form.input_units.currentText()
            factor = 1.0 if units == 'inch' else 2.54
            return factor / float(form.input_dpi.currentText())

        @lock.skipIfCircular
        def update_units(*args):
            width = form.input_width.value()
            height = form.input_height.value()
            factor = get_factor()
            form.input_width_units.setValue(width * factor)
            form.input_height_units.setValue(height * factor)

        @lock.skipIfCircular
        def update_pixels(*args):
            width = form.input_width_units.value()
            height = form.input_height_units.value()
            factor = get_factor()
            form.input_width.setValue(int(width / factor))
            form.input_height.setValue(int(height / factor))

        @lock.skipIfCircular
        def update_width(*args):
            if form.aspectratio > 0:
                width = form.input_height.value() * form.aspectratio
                form.input_width.setValue(int(width))
                form.input_width_units.setValue(width * get_factor())

        @lock.skipIfCircular
        def update_height(*args):
            if form.aspectratio > 0:
                height = form.input_width.value() / form.aspectratio
                form.input_height.setValue(int(height))
                form.input_height_units.setValue(height * get_factor())

        def update_aspectratio(checked=True):
            if checked:
                try:
                    form.aspectratio = (
                            float(form.input_width.value()) /
                            float(form.input_height.value()))
                except ZeroDivisionError:
                    form.button_lock.setChecked(False)
            else:
                form.aspectratio = 0

        def update_from_viewport():
            w, h = self.cmd.get_viewport()
            form.aspectratio = 0
            form.input_width.setValue(w)
            form.input_height.setValue(h)
            update_aspectratio(form.button_lock.isChecked())

        def run_draw(ray=False):
            width = form.input_width.value()
            height = form.input_height.value()
            if ray:
                self.cmd.set('opaque_background',
                        not form.input_transparent.isChecked())
                self.cmd.do('ray %d, %d, async=1' % (width, height))
            else:
                self.cmd.do('draw %d, %d' % (width, height))
            form.stack.setCurrentIndex(1)

        def run_ray():
            run_draw(ray=True)

        def run_save():
            fname = getSaveFileNameWithExt(self, 'Save As...', self.initialdir,
                    filter='PNG File (*.png)')
            if not fname:
                return
            self.initialdir = os.path.dirname(fname)
            self.cmd.png(fname, prior=1, dpi=form.input_dpi.currentText())

        def run_copy_clipboard():
            with PopupOnException():
                _copy_image(self.cmd, False, form.input_dpi.currentText())

        dpi = self.cmd.get_setting_int('image_dots_per_inch')
        if dpi > 0:
            form.input_dpi.setEditText(str(dpi))
        form.input_dpi.setValidator(QtGui.QIntValidator())

        # This connection used to be in the .ui file, but that fails with Qt6
        form.input_units.currentTextChanged.connect(lambda s: form.input_height_units.setSuffix(s))
        form.input_units.currentTextChanged.connect(lambda s: form.input_width_units.setSuffix(s))

        form.input_units.currentIndexChanged.connect(update_units)
        form.input_dpi.editTextChanged.connect(update_pixels)
        form.input_width.valueChanged.connect(update_units)
        form.input_height.valueChanged.connect(update_units)
        form.input_width_units.valueChanged.connect(update_pixels)
        form.input_height_units.valueChanged.connect(update_pixels)

        # set values before connecting mutual width<->height updates
        update_from_viewport()

        form.input_width.valueChanged.connect(update_height)
        form.input_height.valueChanged.connect(update_width)
        form.input_width_units.valueChanged.connect(update_height)
        form.input_height_units.valueChanged.connect(update_width)
        form.button_lock.toggled.connect(update_aspectratio)

        form.button_draw.clicked.connect(run_draw)
        form.button_ray.clicked.connect(run_ray)
        form.button_current.clicked.connect(update_from_viewport)
        form.button_back.clicked.connect(lambda: form.stack.setCurrentIndex(0))
        form.button_clip.clicked.connect(run_copy_clipboard)
        form.button_save.clicked.connect(run_save)

        if widget is None:
            form._dialog.show()
        return form

    @PopupOnException.decorator
    def _file_save(self, filter, format):
        fname = getSaveFileNameWithExt(
            self,
            'Save As...',
            self.initialdir,
            filter=filter)
        if fname:
            self.cmd.save(fname, format=format, quiet=0)

    def file_save_wrl(self):
        self._file_save('VRML 2 WRL File (*.wrl)', 'wrl')

    def file_save_dae(self):
        self._file_save('COLLADA File (*.dae)', 'dae')

    def file_save_pov(self):
        self._file_save('POV File (*.pov)', 'pov')

    def file_save_mpng(self):
        self.file_save_mpeg('png')

    def file_save_mov(self):
        self.file_save_mpeg('mov')

    def file_save_stl(self):
        self._file_save('STL File (*.stl)', 'stl')

    def file_save_gltf(self):
        self._file_save('GLTF File (*.gltf)', 'gltf')

    LOG_FORMATS = [
        'PyMOL Script (*.pml)',
        'Python Script (*.py *.pym)',
        'All (*)',
    ]

    def log_open(self, fname='', mode='w'):
        if not fname:
            fname = getSaveFileNameWithExt(self, 'Open Logfile...', self.initialdir,
                                    filter=';;'.join(self.LOG_FORMATS))
        if fname:
            self.initialdir = os.path.dirname(fname)
            self.cmd.log_open(fname, mode)

    def log_append(self):
        return self.log_open(mode='a')

    def log_resume(self):
        fname = getSaveFileNameWithExt(self, 'Open Logfile...', self.initialdir,
                                filter=';;'.join(self.LOG_FORMATS))
        if fname:
            self.initialdir = os.path.dirname(fname)
            self.cmd.resume(fname)

    def file_run(self):
        formats = [
            'All Runnable (*.pml *.py *.pym)',
            'PyMOL Command Script (*.pml)',
            'PyMOL Command Script (*.txt)',
            'Python Script (*.py *.pym)',
            'Python Script (*.txt)',
            'All Files(*)',
        ]
        fnames, selectedfilter = getOpenFileNames(
            self, 'Open file', self.initialdir, filter=';;'.join(formats))
        is_py = selectedfilter.startswith('Python')

        with PopupOnException():
            for fname in fnames:
                self.initialdir = os.path.dirname(fname)
                self.cmd.cd(self.initialdir, quiet=0)
                # detect: .py, .pym, .pyc, .pyo, .py.txt
                if is_py or re.search(r'\.py(|m|c|o|\.txt)$', fname, re.I):
                    self.cmd.run(fname)
                else:
                    self.cmd.do("@" + fname)

    def cd_dialog(self):
        dname = QFileDialog.getExistingDirectory(
            self, "Change Working Directory", self.initialdir)
        self.cmd.cd(dname or '.', quiet=0)

    def confirm_quit(self):
        QtWidgets.QApplication.instance().quit()

    def settings_edit_all_dialog(self):
        from .advanced_settings_gui import PyMOLAdvancedSettings
        if self.advanced_settings_dialog is None:
            self.advanced_settings_dialog = PyMOLAdvancedSettings(self,
                                                                  self.cmd)
        self.advanced_settings_dialog.show()

    def shortcut_menu_edit_dialog(self):
        from .shortcut_menu_gui import PyMOLShortcutMenu
        if self.shortcut_menu_filter_dialog is None:
            self.shortcut_menu_filter_dialog = PyMOLShortcutMenu(self, self.saved_shortcuts, self.cmd)
        self.shortcut_menu_filter_dialog.show()

    def scene_panel_menu_dialog(self):
        from .scene_bin_gui import ScenePanel

        if self.scene_panel_dialog is None:
            self.scene_panel_dialog = ScenePanel(self)

        self.scene_panel_dialog.show()

    def open_ai_panel(self):
        from pmg_qt.ai_panel import AIPanel

        if self.ai_panel is None:
            self.ai_panel = AIPanel(self)
            self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.ai_panel)

        self.ai_panel.show()
        self.ai_panel.raise_()

    def open_ai_settings(self):
        self.open_ai_panel()
        self.ai_panel.open_settings()

    def _setup_casp_toolbar(self, cmd):
        from pymol import casp17

        toolbar = self.addToolBar('CASP17')
        toolbar.setObjectName('casp17_toolbar')

        mode_action = None

        def mode_command():
            value = 1 if mode_action.isChecked() else 0
            cmd.set('casp_submission_mode', value, log=1, quiet=0)
            self._handle_casp_mode_changed(value)

        mode_action = SettingAction(
            self, cmd, 'casp_submission_mode', 'CASP Mode',
            command=mode_command)
        mode_action.setToolTip(
            'Build an explicit ranked CASP17 model list with object-panel R buttons')
        toolbar.addAction(mode_action)

        rank_widget = QtWidgets.QWidget(self)
        rank_widget.setObjectName('casp_rank_widget')
        rank_layout = QtWidgets.QHBoxLayout(rank_widget)
        rank_layout.setContentsMargins(4, 0, 4, 0)
        rank_layout.setSpacing(2)
        self.casp_rank_labels = []
        self.casp_rank_clear_buttons = []
        for index in range(1, casp17.RANK_COUNT + 1):
            slot_widget = QtWidgets.QWidget(self)
            slot_widget.setObjectName('casp_rank_slot_%d' % index)
            slot_layout = QtWidgets.QHBoxLayout(slot_widget)
            slot_layout.setContentsMargins(0, 0, 0, 0)
            slot_layout.setSpacing(1)

            label = QtWidgets.QLabel('%d. ' % index, slot_widget)
            label.setObjectName('casp_rank_%d' % index)
            label.setMinimumWidth(70)
            label.setMaximumWidth(140)
            label.setStyleSheet(
                'QLabel { background: #d32323; color: white; padding: 2px 5px;'
                ' border-radius: 2px; }')
            slot_layout.addWidget(label)
            self.casp_rank_labels.append(label)

            clear_button = QtWidgets.QToolButton(slot_widget)
            clear_button.setObjectName('casp_clear_rank_%d' % index)
            clear_button.setText('x')
            clear_button.setFixedWidth(18)
            clear_button.setToolTip('Clear CASP rank %d' % index)
            clear_button.setStyleSheet(
                'QToolButton { background: #7f1717; color: white; padding: 0;'
                ' border: 0; }'
                'QToolButton:disabled { background: #4d2a2a; color: #999; }')
            clear_button.clicked.connect(
                lambda _checked=False, rank=index: self.clear_casp_rank(rank))
            slot_layout.addWidget(clear_button)
            self.casp_rank_clear_buttons.append(clear_button)

            rank_layout.addWidget(slot_widget)
        toolbar.addWidget(rank_widget)

        preview_action = QtWidgets.QAction('Preview Selection', self)
        preview_action.setToolTip('Show the ranked CASP17 models in grid view')
        preview_action.triggered.connect(self.toggle_casp_preview)
        toolbar.addAction(preview_action)

        group_action = QtWidgets.QAction('Group ID...', self)
        group_action.setToolTip('Configure the CASP group ID used for submissions')
        group_action.triggered.connect(self.configure_casp_group_id)
        toolbar.addAction(group_action)

        submit_action = QtWidgets.QAction('Submit to CASP', self)
        submit_action.setToolTip(
            'Submit the selected ranked molecule objects to CASP17')
        submit_action.triggered.connect(self.open_casp_submission_dialog)
        toolbar.addAction(submit_action)

        index = cmd.setting._get_index('casp_submission_mode')
        self.setting_callbacks[index].append(self._handle_casp_mode_changed)

        self.casp_toolbar = toolbar
        self.casp_mode_action = mode_action
        self.casp_rank_widget = rank_widget
        self.casp_preview_action = preview_action
        self.casp_group_action = group_action
        self.casp_submit_action = submit_action
        self._handle_casp_mode_changed(cmd.get_setting_int('casp_submission_mode'))

    def _handle_casp_mode_changed(self, value):
        from pymol import casp17

        enabled = bool(int(value))
        casp17.handle_mode_change(enabled, self.cmd)

        if not enabled:
            self._casp_pending_model_name = None
            if self._casp_restore_state is not None:
                self._restore_casp_preview()

        for label in self.casp_rank_labels:
            label.setVisible(enabled)
        for button in self.casp_rank_clear_buttons:
            button.setVisible(enabled)
        if hasattr(self, 'casp_rank_widget'):
            self.casp_rank_widget.setVisible(enabled)
        if hasattr(self, 'casp_preview_action'):
            self.casp_preview_action.setVisible(enabled)
        self._update_casp_rank_widgets()

    def _update_casp_rank_widgets(self):
        from pymol import casp17

        enabled = bool(self.cmd.get_setting_int('casp_submission_mode'))
        ranked_model_names = casp17.get_ranked_model_names(self.cmd)
        complete = enabled and all(ranked_model_names)
        has_selection = enabled and any(ranked_model_names)

        for index, label in enumerate(self.casp_rank_labels, start=1):
            name = ranked_model_names[index - 1]
            label.setText('%d. %s' % (index, name or ''))
            label.setToolTip(name or '')
            if index <= len(self.casp_rank_clear_buttons):
                self.casp_rank_clear_buttons[index - 1].setEnabled(
                    enabled and bool(name))

        if self._casp_restore_state is not None and not has_selection:
            self._restore_casp_preview()
            return

        if hasattr(self, 'casp_preview_action'):
            self.casp_preview_action.setEnabled(
                has_selection or self._casp_restore_state is not None)
            self.casp_preview_action.setText(
                'Restore View' if self._casp_restore_state is not None
                else 'Preview Selection')

        if hasattr(self, 'casp_submit_action'):
            self.casp_submit_action.setEnabled(complete)

    def handle_casp_object_panel_pick(self, model_name):
        from pymol import casp17

        if not self.cmd.get_setting_int('casp_submission_mode'):
            return None
        if not casp17.is_valid_model_name(model_name, self.cmd):
            return None

        self._casp_pending_model_name = model_name
        self.pymolwidget.setFocus()
        self.statusBar().showMessage(
            'CASP rank pending: %s. Press 1-5 to assign, 0/Delete to remove.' %
            model_name, 5000)
        return model_name

    def handle_casp_rank_key(self, ev):
        if not self._casp_pending_model_name:
            return False

        if ev.key() == Qt.Key.Key_Escape:
            self._casp_pending_model_name = None
            self.statusBar().clearMessage()
            return True

        text = ev.text()
        clear_keys = (
            Qt.Key.Key_Delete,
            Qt.Key.Key_Backspace,
        )
        if text == '0' or ev.key() in clear_keys:
            from pymol import casp17

            model_name = self._casp_pending_model_name
            self._casp_pending_model_name = None
            self.statusBar().clearMessage()
            casp17.clear_ranked_model_name(model_name, self.cmd)
            self._handle_casp_rank_list_changed()
            return True

        if text in ('1', '2', '3', '4', '5'):
            from pymol import casp17

            model_name = self._casp_pending_model_name
            self._casp_pending_model_name = None
            self.statusBar().clearMessage()
            try:
                casp17.assign_ranked_model(int(text), model_name, self.cmd)
                self._handle_casp_rank_list_changed()
            except casp17.CASPSubmissionError as e:
                QtWidgets.QMessageBox.warning(
                    self, 'CASP17 Selection', str(e))
            return True

        self._casp_pending_model_name = None
        self.statusBar().clearMessage()
        return False

    def _handle_casp_rank_list_changed(self):
        from pymol import casp17

        ranked_model_names = casp17.get_ranked_model_names(self.cmd)
        if self._casp_restore_state is not None:
            if any(ranked_model_names):
                self._refresh_casp_preview()
            else:
                self._restore_casp_preview()
                return
        self._update_casp_rank_widgets()

    @PopupOnException.decorator
    def clear_casp_rank(self, rank):
        from pymol import casp17

        casp17.clear_ranked_model_rank(rank, self.cmd)
        self._handle_casp_rank_list_changed()

    @PopupOnException.decorator
    def toggle_casp_preview(self, _checked=False):
        from pymol import casp17

        if self._casp_restore_state is not None:
            self._restore_casp_preview()
            return

        try:
            model_names = casp17.get_ranked_model_names(self.cmd)
            self._casp_restore_state = casp17.prepare_grid_preview(
                model_names, self.cmd)
            self._set_casp_preview_border(True)
        except casp17.CASPSubmissionError as e:
            QtWidgets.QMessageBox.warning(self, 'CASP17 Preview', str(e))
            return

        self._update_casp_rank_widgets()

    def _refresh_casp_preview(self):
        from pymol import casp17

        if self._casp_restore_state is None:
            return
        model_names = casp17.get_ranked_model_names(self.cmd)
        casp17.apply_grid_preview(model_names, self.cmd)

    def _set_casp_preview_border(self, enabled):
        set_border = getattr(self.pymolwidget, 'set_casp_preview_border', None)
        if set_border is not None:
            set_border(enabled)

    @PopupOnException.decorator
    def configure_casp_group_id(self, _checked=False):
        from pymol import casp17

        current_group_id = casp17.get_group_id(self.cmd)
        group_id, accepted = QtWidgets.QInputDialog.getText(
            self,
            'CASP17 Group ID',
            'Group ID',
            text=current_group_id)
        if not accepted:
            return

        try:
            casp17.set_group_id(group_id, self.cmd)
        except casp17.CASPSubmissionError as e:
            QtWidgets.QMessageBox.warning(self, 'CASP17 Group ID', str(e))

    @PopupOnException.decorator
    def open_casp_submission_dialog(self, _checked=False):
        from pymol import casp17

        try:
            model_names = casp17.get_validated_ranked_model_names(self.cmd)
        except casp17.CASPSubmissionError as e:
            QtWidgets.QMessageBox.warning(self, 'CASP17 Submission', str(e))
            return

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle('Submit to CASP17')
        layout = QtWidgets.QVBoxLayout(dialog)

        form = QtWidgets.QFormLayout()
        target_input = QtWidgets.QLineEdit()
        target_input.setPlaceholderText('H1312')
        group_input = QtWidgets.QLineEdit()
        group_input.setText(casp17.get_group_id(self.cmd))
        group_input.setPlaceholderText('Required')
        email_input = QtWidgets.QLineEdit()
        form.addRow('Target ID', target_input)
        form.addRow('Group ID', group_input)
        form.addRow('Email', email_input)
        layout.addLayout(form)

        model_list = QtWidgets.QListWidget()
        model_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        for index, name in enumerate(model_names[:5], start=1):
            model_list.addItem('%d. %s' % (index, name))
        layout.addWidget(model_list)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok |
            QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.button(
            QtWidgets.QDialogButtonBox.StandardButton.Ok).setText('Submit')
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return

        target_id = target_input.text()
        group_id = group_input.text()
        email = email_input.text()

        QtWidgets.QApplication.setOverrideCursor(
            Qt.CursorShape.WaitCursor)
        try:
            group_id = casp17.set_group_id(group_id, self.cmd)
            result = casp17.build_current_submission(
                target_id, group_id=group_id, _self=self.cmd)
            result['response_text'] = casp17.submit_to_casp(
                result['output_path'], email)
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, 'CASP17 Submission Failed', str(e))
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        self._show_casp_submission_success(result)

    def _show_casp_submission_success(self, result):
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle('CASP17 Submission')
        box.setIcon(QtWidgets.QMessageBox.Icon.Information)
        box.setText('CASP17 submission complete.')
        model_order = '\n'.join(
            '%d. %s' % (index, name)
            for index, name in enumerate(result['model_names'], start=1))
        box.setInformativeText(
            'Group ID: %s\n\nSaved file:\n%s\n\nSubmitted models:\n%s' %
            (result['group_id'], result['output_path'], model_order))

        response_text = result.get('response_text') or ''
        if response_text:
            box.setDetailedText(response_text)

        restore_button = None
        if self._casp_restore_state is not None:
            restore_button = box.addButton(
                'Restore Previous View',
                QtWidgets.QMessageBox.ButtonRole.ActionRole)
        box.addButton(QtWidgets.QMessageBox.StandardButton.Close)
        box.exec()

        if restore_button is not None and box.clickedButton() is restore_button:
            self._restore_casp_preview()

    def _restore_casp_preview(self):
        from pymol import casp17

        if self._casp_restore_state is not None:
            casp17.restore_grid_preview(self._casp_restore_state, self.cmd)
            self._casp_restore_state = None
            self._set_casp_preview_border(False)
            if hasattr(self, 'casp_preview_action'):
                self.casp_preview_action.setText('Preview Selection')
            self._update_casp_rank_widgets()

    def show_about(self):
        msg = [
            'The PyMOL Molecular Graphics System\n',
            'Version %s' % (self.cmd.get_version()[0]),
            u'Copyright (C) Schr\xF6dinger, LLC.',
            'All rights reserved.\n',
            'License information:',
        ]

        msg.append('Open-Source Build')

        msg += [
            '',
            'For more information:',
            'https://pymol.org',
            'sales@schrodinger.com',
        ]
        QtWidgets.QMessageBox.about(self, "About PyMOL", '\n'.join(msg))

    #################
    # GUI callbacks
    #################

    def command_get(self):
        return self.lineedit.text()

    def command_set(self, v):
        return self.lineedit.setText(v)

    def command_set_cursor(self, i):
        return self.lineedit.setCursorPosition(i)

    def update_progress(self):
        progress = int(self.cmd.get_progress() * 100)
        if progress >= 0:
            self.progressbar.setValue(progress)
            self.progressbar.show()
            self.abortbutton.show()
        else:
            self.progressbar.hide()
            self.abortbutton.hide()

    def update_feedback(self):
        self.update_progress()

        feedback = self.cmd._get_feedback()
        if feedback:
            html = colorprinting.text2html('\n'.join(feedback))
            self.browser.appendHtml(html)

            scrollbar = self.browser.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

        for setting in self.cmd.get_setting_updates() or ():
            if setting in self.setting_callbacks:
                current_value = self.cmd.get_setting_tuple(setting)[1][0]
                for callback in self.setting_callbacks[setting]:
                    callback(current_value)

        self.feedback_timer.start(500)

    def doPrompt(self):
        self.doTypedCommand(self.command_get())
        self.pymolwidget._pymolProcess()
        self.lineedit.clear()
        self.feedback_timer.start(0)

    ##########################
    # legacy plugin system
    ##########################

    @PopupOnException.decorator
    def initializePlugins(self):
        from pymol import plugins
        from . import mimic_tk

        self.menudict['Plugin'].clear()

        app = plugins.get_pmgapp()

        plugins.legacysupport.addPluginManagerMenuItem()

        # Redirect to Legacy submenu
        self.menudict['PluginQt'] = self.menudict['Plugin']
        self.menudict['Plugin'] = self.menudict['PluginQt'].addMenu('Legacy Plugins')
        self.menudict['Plugin'].setTearOffEnabled(True)
        self.menudict['PluginQt'].addSeparator()

        plugins.HAVE_QT = True
        plugins.initialize(app)

    def createlegacypmgapp(self):
        from . import mimic_pmg_tk as mimic
        pmgapp = mimic.PMGApp()
        pmgapp.menuBar = mimic.PmwMenuBar(self.menudict)
        return pmgapp

    def window_cmd(self, action, x, y, w, h):
        if action == 0: # hide
            self.hide()
        elif action == 1: # show
            self.show()
        elif action == 2: # position
            self.move(x, y)
        elif action == 3: # size (first two arguments)
            self.resize(x, y)
        elif action == 4: # box
            self.move(x, y)
            self.resize(w, h)
        elif action == 5: # maximize
            self.showMaximized()
        elif action == 6: # fit
            if hasattr(QtGui, 'QWindow') and self.windowHandle().visibility() in (
                    QtGui.QWindow.Visibility.Maximized, QtGui.QWindow.Visibility.FullScreen):
                return
            a = self.screen().availableGeometry()
            g = self.geometry()
            f = self.frameGeometry()
            w = min(f.width(), a.width())
            h = min(f.height(), a.height())
            x = max(min(f.x(), a.right() - w), a.x())
            y = max(min(f.y(), a.bottom() - h), a.y())
            self.setGeometry(
                x - f.x() + g.x(),
                y - f.y() + g.y(),
                w - f.width() + g.width(),
                h - f.height() + g.height(),
            )
        elif action == 7: # focus
            self.setFocus(Qt.FocusReason.OtherFocusReason)
        elif action == 8: # defocus
            self.clearFocus()


def commandoverloaddecorator(func):
    name = func.__name__
    func.__doc__ = getattr(pymol.cmd, name).__doc__
    setattr(pymol.cmd, name, func)
    pymol.cmd.extend(func)
    return func


def SettingAction(parent, cmd, name, label='', true_value=1, false_value=0,
                  command=None):
    '''
    Menu toggle action for a PyMOL setting

    parent: parent QObject
    cmd: PyMOL instance
    name: setting name
    label: menu item text
    '''
    if not label:
        label = name

    index = cmd.setting._get_index(name)
    type_, values = cmd.get_setting_tuple(index)
    action = QtWidgets.QAction(label, parent)

    if not command:
        command = lambda: cmd.set(
            index,
            true_value if action.isChecked() else false_value,
            log=1,
            quiet=0)

    parent.setting_callbacks[index].append(
        lambda v: action.setChecked(v != false_value))

    if type_ in (
            1,  # bool
            2,  # int
            3,  # float
            5,  # color
            6,  # str
    ):
        action.setCheckable(True)
        if values[0] == true_value:
            action.setChecked(True)
    else:
        print('TODO', type_, name)

    action.triggered.connect(command)
    return action

window = None


class CommandLineEdit(QtWidgets.QLineEdit):
    '''
    Line edit widget with instant text insert on drag-enter
    '''
    _saved_pos = -1

    def dragMoveEvent(self, event):
        pass

    def dropEvent(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()

    def dragEnterEvent(self, event):
        if not event.mimeData().hasText():
            self._saved_pos = -1
            return

        event.acceptProposedAction()

        urls = event.mimeData().urls()
        if urls and urls[0].isLocalFile():
            droppedtext = urls[0].toLocalFile()
        else:
            droppedtext = event.mimeData().text()

        pos = self.cursorPosition()
        text = self.text()
        self._saved_pos = pos
        self._saved_text = text

        self.setText(text[:pos] + droppedtext + text[pos:])
        self.setSelection(pos, len(droppedtext))

    def dragLeaveEvent(self, event):
        if self._saved_pos != -1:
            self.setText(self._saved_text)
            self.setCursorPosition(self._saved_pos)


class PyMOLApplication(QtWidgets.QApplication):
    '''
    Catch drop events on app icon
    '''
    # FileOpen event is only activated after the first
    # application state change, otherwise sys.argv would be
    # handled by Qt, we don't want that.

    def handle_file_open(self, ev):
        if ev.type() == QtCore.QEvent.Type.ApplicationActivate:
            self.handle_file_open = self.handle_file_open_active
        return False

    def handle_file_open_active(self, ev):
        if ev.type() != QtCore.QEvent.Type.FileOpen:
            return False

        # When double clicking a file in Finder, open it in a new instance
        if not pymol.invocation.options.reuse_helper and pymol.cmd.get_names():
            window.new_window([ev.file()])
            return True

        # pymol -I -U
        if pymol.invocation.options.auto_reinitialize:
            pymol.cmd.reinitialize()

        # PyMOL Show
        if ev.file().endswith('.psw'):
            pymol.cmd.set('presentation')
            pymol.cmd.set('internal_gui', 0)
            pymol.cmd.set('internal_feedback', 0)
            pymol.cmd.full_screen('on')

        window.load_dialog(ev.file())
        return True

    def event(self, ev):
        if self.handle_file_open(ev):
            return True
        return super(PyMOLApplication, self).event(ev)


# like pymol.internal._copy_image
def _copy_image(_self=pymol.cmd, quiet=1, dpi=-1):
    import tempfile
    fname = tempfile.mktemp('.png')

    if not _self.png(fname, prior=1, dpi=dpi):
        print("no prior image")
        return

    try:
        qim = QtGui.QImage(fname)
        QtWidgets.QApplication.clipboard().setImage(qim)
    finally:
        os.unlink(fname)

    if not quiet:
        print(" Image copied to clipboard")


def make_pymol_qicon():
    icons_dir = os.path.expandvars('$PYMOL_DATA/pymol/icons')
    return QtGui.QIcon(os.path.join(icons_dir, 'icon2.svg'))


def execapp():
    '''
    Run PyMOL as a Qt application
    '''
    global window
    global pymol

    # don't let exceptions stop PyMOL
    import traceback
    sys.excepthook = traceback.print_exception

    # use QT_OPENGL=desktop (auto-detection may fail on Windows)
    if hasattr(Qt, 'AA_UseDesktopOpenGL') and pymol.IS_WINDOWS:
        QtCore.QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_UseDesktopOpenGL)

    # enable 4K scaling on Windows and Linux
    if hasattr(Qt, 'AA_EnableHighDpiScaling') and not any(
            v in os.environ
            for v in ['QT_SCALE_FACTOR', 'QT_SCREEN_SCALE_FACTORS']):
        QtCore.QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling)

    # fix Windows taskbar icon
    if pymol.IS_WINDOWS:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                u'com.schrodinger.pymol')

    app = PyMOLApplication(['PyMOL'])
    app.setWindowIcon(make_pymol_qicon())

    window = PyMOLQtGUI()
    window.setWindowTitle("PyMOL")

    # fix gnome/wayland dash icon/missing wmclass
    app.setDesktopFileName("org.pymol.PyMOL")

    @commandoverloaddecorator
    def viewport(w=-1, h=-1, _self=None):
        window.viewportsignal.emit(int(w), int(h))

    @commandoverloaddecorator
    def full_screen(toggle=-1, _self=None):
        from pymol import viewing as v
        toggle = v.toggle_dict[v.toggle_sc.auto_err(str(toggle), 'toggle')]
        window.toggle_fullscreen(toggle)

    import pymol.gui
    pymol.gui.createlegacypmgapp = window.createlegacypmgapp

    pymol.cmd._copy_image = _copy_image
    pymol.cmd._call_in_gui_thread = MainThreadCaller()

    # Assume GUI thread, make OpenGL context current before calling func().
    def _call_with_opengl_context_gui_thread(func):
        with window.pymolwidget:
            return func()

    # Dispatch to GUI thread and make OpenGL context current before calling func().
    pymol.cmd._call_with_opengl_context = lambda func: pymol.cmd._call_in_gui_thread(
        lambda: _call_with_opengl_context_gui_thread(func))

    window.show()
    window.raise_()

    # window size according to -W -H options
    options = pymol.invocation.options
    if options.win_xy_set:
        scale = window.pymolwidget.fb_scale
        viewport(scale * options.win_x, scale * options.win_y)

    # load plugins
    if options.plugins:
        window.initializePlugins()

    app.exec()
