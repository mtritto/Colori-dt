# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'ColorI-DT.ui'
##
## Created by: Qt User Interface Compiler version 6.5.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient,
    QTransform)
from PySide6.QtWidgets import (QApplication, QComboBox, QDoubleSpinBox, QFrame,
    QGraphicsView, QHBoxLayout, QLabel, QMainWindow,
    QMenu, QMenuBar, QProgressBar, QPushButton,
    QSizePolicy, QSpinBox, QTabWidget, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1200, 700)
        sizePolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMinimumSize(QSize(1200, 700))
        MainWindow.setMaximumSize(QSize(1200, 719))
        MainWindow.setBaseSize(QSize(1000, 700))
        MainWindow.setDockOptions(QMainWindow.AllowTabbedDocks|QMainWindow.AnimatedDocks|QMainWindow.VerticalTabs)
        self.actionExit = QAction(MainWindow)
        self.actionExit.setObjectName(u"actionExit")
        self.actionHelp = QAction(MainWindow)
        self.actionHelp.setObjectName(u"actionHelp")
        self.actionAbout = QAction(MainWindow)
        self.actionAbout.setObjectName(u"actionAbout")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.horizontalLayout_3 = QHBoxLayout(self.centralwidget)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.left_frame = QFrame(self.centralwidget)
        self.left_frame.setObjectName(u"left_frame")
        sizePolicy.setHeightForWidth(self.left_frame.sizePolicy().hasHeightForWidth())
        self.left_frame.setSizePolicy(sizePolicy)
        self.left_frame.setMinimumSize(QSize(350, 650))
        font = QFont()
        font.setBold(False)
        self.left_frame.setFont(font)
        self.left_frame.setFrameShape(QFrame.StyledPanel)
        self.left_frame.setFrameShadow(QFrame.Raised)
        self.gv_ref_img = QGraphicsView(self.left_frame)
        self.gv_ref_img.setObjectName(u"gv_ref_img")
        self.gv_ref_img.setEnabled(True)
        self.gv_ref_img.setGeometry(QRect(30, 40, 191, 181))
        self.gv_ref_img.setAutoFillBackground(True)
        self.gv_ref_img.setFrameShape(QFrame.StyledPanel)
        self.gv_ref_img.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.gv_ref_img.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.pb_ref_select = QPushButton(self.left_frame)
        self.pb_ref_select.setObjectName(u"pb_ref_select")
        self.pb_ref_select.setGeometry(QRect(230, 100, 113, 32))
        self.pb_ref_reset = QPushButton(self.left_frame)
        self.pb_ref_reset.setObjectName(u"pb_ref_reset")
        self.pb_ref_reset.setGeometry(QRect(230, 130, 113, 32))
        self.pb_test_select = QPushButton(self.left_frame)
        self.pb_test_select.setObjectName(u"pb_test_select")
        self.pb_test_select.setGeometry(QRect(230, 430, 113, 32))
        self.pb_test_reset = QPushButton(self.left_frame)
        self.pb_test_reset.setObjectName(u"pb_test_reset")
        self.pb_test_reset.setGeometry(QRect(230, 460, 113, 32))
        self.line = QFrame(self.left_frame)
        self.line.setObjectName(u"line")
        self.line.setGeometry(QRect(7, 310, 331, 20))
        self.line.setFrameShape(QFrame.HLine)
        self.line.setFrameShadow(QFrame.Sunken)
        self.label = QLabel(self.left_frame)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(30, 20, 111, 16))
        font1 = QFont()
        font1.setPointSize(13)
        font1.setBold(True)
        font1.setUnderline(False)
        font1.setStrikeOut(False)
        font1.setKerning(True)
        self.label.setFont(font1)
        self.label_2 = QLabel(self.left_frame)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(30, 340, 111, 16))
        font2 = QFont()
        font2.setBold(True)
        self.label_2.setFont(font2)
        self.gv_test_img = QGraphicsView(self.left_frame)
        self.gv_test_img.setObjectName(u"gv_test_img")
        self.gv_test_img.setEnabled(True)
        self.gv_test_img.setGeometry(QRect(30, 360, 191, 181))
        self.gv_test_img.setAutoFillBackground(True)
        self.gv_test_img.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.gv_test_img.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.label_26 = QLabel(self.left_frame)
        self.label_26.setObjectName(u"label_26")
        self.label_26.setGeometry(QRect(30, 560, 61, 16))
        self.label_27 = QLabel(self.left_frame)
        self.label_27.setObjectName(u"label_27")
        self.label_27.setGeometry(QRect(30, 590, 61, 16))
        self.label_28 = QLabel(self.left_frame)
        self.label_28.setObjectName(u"label_28")
        self.label_28.setGeometry(QRect(30, 240, 61, 16))
        self.label_29 = QLabel(self.left_frame)
        self.label_29.setObjectName(u"label_29")
        self.label_29.setGeometry(QRect(30, 270, 61, 16))
        self.la_test_h = QLabel(self.left_frame)
        self.la_test_h.setObjectName(u"la_test_h")
        self.la_test_h.setGeometry(QRect(130, 560, 60, 16))
        self.la_test_w = QLabel(self.left_frame)
        self.la_test_w.setObjectName(u"la_test_w")
        self.la_test_w.setGeometry(QRect(130, 590, 60, 16))
        self.la_ref_h = QLabel(self.left_frame)
        self.la_ref_h.setObjectName(u"la_ref_h")
        self.la_ref_h.setGeometry(QRect(130, 240, 60, 16))
        self.la_ref_w = QLabel(self.left_frame)
        self.la_ref_w.setObjectName(u"la_ref_w")
        self.la_ref_w.setGeometry(QRect(130, 270, 60, 16))

        self.horizontalLayout_3.addWidget(self.left_frame)

        self.tabWidget = QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName(u"tabWidget")
        self.tabWidget.setEnabled(True)
        sizePolicy1 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.tabWidget.sizePolicy().hasHeightForWidth())
        self.tabWidget.setSizePolicy(sizePolicy1)
        self.tabWidget.setFont(font)
        self.tabWidget.setLayoutDirection(Qt.LeftToRight)
        self.tabWidget.setAutoFillBackground(False)
        self.tabWidget.setTabPosition(QTabWidget.North)
        self.tabWidget.setTabShape(QTabWidget.Rounded)
        self.tabWidget.setElideMode(Qt.ElideLeft)
        self.tabWidget.setUsesScrollButtons(False)
        self.tabWidget.setTabBarAutoHide(False)
        self.tab_metrics = QWidget()
        self.tab_metrics.setObjectName(u"tab_metrics")
        self.gv_metrics_out = QGraphicsView(self.tab_metrics)
        self.gv_metrics_out.setObjectName(u"gv_metrics_out")
        self.gv_metrics_out.setEnabled(True)
        self.gv_metrics_out.setGeometry(QRect(160, 260, 371, 331))
        self.gv_metrics_out.setAutoFillBackground(True)
        self.gv_metrics_out.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.gv_metrics_out.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.line_2 = QFrame(self.tab_metrics)
        self.line_2.setObjectName(u"line_2")
        self.line_2.setGeometry(QRect(10, 210, 781, 20))
        self.line_2.setFont(font)
        self.line_2.setFrameShape(QFrame.HLine)
        self.line_2.setFrameShadow(QFrame.Sunken)
        self.label_3 = QLabel(self.tab_metrics)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setGeometry(QRect(160, 240, 171, 16))
        self.label_3.setFont(font2)
        self.pb_diff_calc = QPushButton(self.tab_metrics)
        self.pb_diff_calc.setObjectName(u"pb_diff_calc")
        self.pb_diff_calc.setGeometry(QRect(560, 370, 151, 32))
        self.pb_diff_reset = QPushButton(self.tab_metrics)
        self.pb_diff_reset.setObjectName(u"pb_diff_reset")
        self.pb_diff_reset.setGeometry(QRect(560, 400, 151, 32))
        self.pb_export_diff = QPushButton(self.tab_metrics)
        self.pb_export_diff.setObjectName(u"pb_export_diff")
        self.pb_export_diff.setGeometry(QRect(560, 430, 151, 32))
        self.tab_select_metric = QTabWidget(self.tab_metrics)
        self.tab_select_metric.setObjectName(u"tab_select_metric")
        self.tab_select_metric.setGeometry(QRect(10, 9, 781, 201))
        sizePolicy.setHeightForWidth(self.tab_select_metric.sizePolicy().hasHeightForWidth())
        self.tab_select_metric.setSizePolicy(sizePolicy)
        self.tab_select_metric.setMaximumSize(QSize(16777215, 250))
        font3 = QFont()
        font3.setBold(False)
        font3.setKerning(False)
        self.tab_select_metric.setFont(font3)
        self.tab_select_metric.setFocusPolicy(Qt.TabFocus)
        self.tab_select_metric.setAutoFillBackground(True)
        self.tab_select_metric.setTabPosition(QTabWidget.North)
        self.tab_select_metric.setTabShape(QTabWidget.Triangular)
        self.tab_select_metric.setElideMode(Qt.ElideNone)
        self.tab_select_metric.setUsesScrollButtons(True)
        self.tab_select_metric.setDocumentMode(False)
        self.tab_select_metric.setTabBarAutoHide(False)
        self.tab_dergb = QWidget()
        self.tab_dergb.setObjectName(u"tab_dergb")
        self.label_6 = QLabel(self.tab_dergb)
        self.label_6.setObjectName(u"label_6")
        self.label_6.setGeometry(QRect(20, 10, 131, 16))
        self.label_14 = QLabel(self.tab_dergb)
        self.label_14.setObjectName(u"label_14")
        self.label_14.setGeometry(QRect(20, 50, 311, 16))
        self.pb_about_rgb = QPushButton(self.tab_dergb)
        self.pb_about_rgb.setObjectName(u"pb_about_rgb")
        self.pb_about_rgb.setGeometry(QRect(10, 130, 131, 31))
        self.tab_select_metric.addTab(self.tab_dergb, "")
        self.tab_cie76lab = QWidget()
        self.tab_cie76lab.setObjectName(u"tab_cie76lab")
        self.label_7 = QLabel(self.tab_cie76lab)
        self.label_7.setObjectName(u"label_7")
        self.label_7.setGeometry(QRect(20, 10, 131, 16))
        self.label_15 = QLabel(self.tab_cie76lab)
        self.label_15.setObjectName(u"label_15")
        self.label_15.setGeometry(QRect(20, 50, 311, 16))
        self.pb_about_76_lab = QPushButton(self.tab_cie76lab)
        self.pb_about_76_lab.setObjectName(u"pb_about_76_lab")
        self.pb_about_76_lab.setGeometry(QRect(10, 130, 131, 31))
        self.tab_select_metric.addTab(self.tab_cie76lab, "")
        self.tab_cie94lab = QWidget()
        self.tab_cie94lab.setObjectName(u"tab_cie94lab")
        self.sb_kL_94 = QDoubleSpinBox(self.tab_cie94lab)
        self.sb_kL_94.setObjectName(u"sb_kL_94")
        self.sb_kL_94.setGeometry(QRect(20, 70, 71, 24))
        self.sb_kL_94.setDecimals(1)
        self.sb_kL_94.setMinimum(0.100000000000000)
        self.sb_kL_94.setSingleStep(0.100000000000000)
        self.sb_kL_94.setValue(1.000000000000000)
        self.sb_k1_94 = QDoubleSpinBox(self.tab_cie94lab)
        self.sb_k1_94.setObjectName(u"sb_k1_94")
        self.sb_k1_94.setGeometry(QRect(120, 70, 71, 24))
        self.sb_k1_94.setDecimals(3)
        self.sb_k1_94.setMinimum(0.001000000000000)
        self.sb_k1_94.setValue(0.100000000000000)
        self.sb_k2_94 = QDoubleSpinBox(self.tab_cie94lab)
        self.sb_k2_94.setObjectName(u"sb_k2_94")
        self.sb_k2_94.setGeometry(QRect(220, 70, 71, 24))
        self.sb_k2_94.setDecimals(3)
        self.sb_k2_94.setMinimum(0.001000000000000)
        self.sb_k2_94.setValue(1.000000000000000)
        self.label_8 = QLabel(self.tab_cie94lab)
        self.label_8.setObjectName(u"label_8")
        self.label_8.setGeometry(QRect(20, 10, 131, 16))
        self.label_18 = QLabel(self.tab_cie94lab)
        self.label_18.setObjectName(u"label_18")
        self.label_18.setGeometry(QRect(20, 50, 21, 16))
        self.label_19 = QLabel(self.tab_cie94lab)
        self.label_19.setObjectName(u"label_19")
        self.label_19.setGeometry(QRect(120, 50, 21, 16))
        self.label_20 = QLabel(self.tab_cie94lab)
        self.label_20.setObjectName(u"label_20")
        self.label_20.setGeometry(QRect(220, 50, 21, 16))
        self.pb_about_94_lab = QPushButton(self.tab_cie94lab)
        self.pb_about_94_lab.setObjectName(u"pb_about_94_lab")
        self.pb_about_94_lab.setGeometry(QRect(10, 130, 131, 31))
        self.tab_select_metric.addTab(self.tab_cie94lab, "")
        self.tab_cie00lab = QWidget()
        self.tab_cie00lab.setObjectName(u"tab_cie00lab")
        self.label_9 = QLabel(self.tab_cie00lab)
        self.label_9.setObjectName(u"label_9")
        self.label_9.setGeometry(QRect(20, 10, 131, 16))
        self.label_21 = QLabel(self.tab_cie00lab)
        self.label_21.setObjectName(u"label_21")
        self.label_21.setGeometry(QRect(20, 50, 21, 16))
        self.label_22 = QLabel(self.tab_cie00lab)
        self.label_22.setObjectName(u"label_22")
        self.label_22.setGeometry(QRect(120, 50, 21, 16))
        self.sb_k2_00 = QDoubleSpinBox(self.tab_cie00lab)
        self.sb_k2_00.setObjectName(u"sb_k2_00")
        self.sb_k2_00.setGeometry(QRect(220, 70, 71, 24))
        self.sb_k2_00.setDecimals(3)
        self.sb_k2_00.setMinimum(0.001000000000000)
        self.sb_k2_00.setSingleStep(0.001000000000000)
        self.sb_k2_00.setValue(0.015000000000000)
        self.sb_kL_00 = QDoubleSpinBox(self.tab_cie00lab)
        self.sb_kL_00.setObjectName(u"sb_kL_00")
        self.sb_kL_00.setGeometry(QRect(20, 70, 71, 24))
        self.sb_kL_00.setDecimals(1)
        self.sb_kL_00.setMinimum(0.100000000000000)
        self.sb_kL_00.setSingleStep(0.100000000000000)
        self.sb_kL_00.setValue(1.000000000000000)
        self.label_23 = QLabel(self.tab_cie00lab)
        self.label_23.setObjectName(u"label_23")
        self.label_23.setGeometry(QRect(220, 50, 21, 16))
        self.sb_k1_00 = QDoubleSpinBox(self.tab_cie00lab)
        self.sb_k1_00.setObjectName(u"sb_k1_00")
        self.sb_k1_00.setGeometry(QRect(120, 70, 71, 24))
        self.sb_k1_00.setDecimals(3)
        self.sb_k1_00.setValue(0.045000000000000)
        self.pb_about_00_lab = QPushButton(self.tab_cie00lab)
        self.pb_about_00_lab.setObjectName(u"pb_about_00_lab")
        self.pb_about_00_lab.setGeometry(QRect(10, 130, 131, 31))
        self.tab_select_metric.addTab(self.tab_cie00lab, "")
        self.tab_cie76luv = QWidget()
        self.tab_cie76luv.setObjectName(u"tab_cie76luv")
        self.label_10 = QLabel(self.tab_cie76luv)
        self.label_10.setObjectName(u"label_10")
        self.label_10.setGeometry(QRect(20, 10, 131, 16))
        self.label_16 = QLabel(self.tab_cie76luv)
        self.label_16.setObjectName(u"label_16")
        self.label_16.setGeometry(QRect(20, 50, 311, 16))
        self.pb_about_76_luv = QPushButton(self.tab_cie76luv)
        self.pb_about_76_luv.setObjectName(u"pb_about_76_luv")
        self.pb_about_76_luv.setGeometry(QRect(10, 130, 131, 31))
        self.tab_select_metric.addTab(self.tab_cie76luv, "")
        self.tab_icsm = QWidget()
        self.tab_icsm.setObjectName(u"tab_icsm")
        self.sb_ang = QSpinBox(self.tab_icsm)
        self.sb_ang.setObjectName(u"sb_ang")
        self.sb_ang.setGeometry(QRect(20, 70, 48, 24))
        self.label_11 = QLabel(self.tab_icsm)
        self.label_11.setObjectName(u"label_11")
        self.label_11.setGeometry(QRect(20, 10, 131, 16))
        self.label_17 = QLabel(self.tab_icsm)
        self.label_17.setObjectName(u"label_17")
        self.label_17.setGeometry(QRect(20, 50, 121, 16))
        self.pb_about_icsm_luv = QPushButton(self.tab_icsm)
        self.pb_about_icsm_luv.setObjectName(u"pb_about_icsm_luv")
        self.pb_about_icsm_luv.setGeometry(QRect(10, 130, 131, 31))
        self.tab_select_metric.addTab(self.tab_icsm, "")
        self.tab_cmc = QWidget()
        self.tab_cmc.setObjectName(u"tab_cmc")
        self.label_12 = QLabel(self.tab_cmc)
        self.label_12.setObjectName(u"label_12")
        self.label_12.setGeometry(QRect(20, 10, 131, 16))
        self.label_13 = QLabel(self.tab_cmc)
        self.label_13.setObjectName(u"label_13")
        self.label_13.setGeometry(QRect(20, 50, 131, 16))
        self.pb_about_cmc_lab = QPushButton(self.tab_cmc)
        self.pb_about_cmc_lab.setObjectName(u"pb_about_cmc_lab")
        self.pb_about_cmc_lab.setGeometry(QRect(10, 130, 131, 31))
        self.cb_ratio = QComboBox(self.tab_cmc)
        self.cb_ratio.addItem("")
        self.cb_ratio.addItem("")
        self.cb_ratio.setObjectName(u"cb_ratio")
        self.cb_ratio.setGeometry(QRect(20, 80, 104, 26))
        self.tab_select_metric.addTab(self.tab_cmc, "")
        self.cb_metrics = QComboBox(self.tab_metrics)
        self.cb_metrics.addItem("")
        self.cb_metrics.addItem("")
        self.cb_metrics.addItem("")
        self.cb_metrics.addItem("")
        self.cb_metrics.addItem("")
        self.cb_metrics.addItem("")
        self.cb_metrics.addItem("")
        self.cb_metrics.setObjectName(u"cb_metrics")
        self.cb_metrics.setGeometry(QRect(565, 290, 141, 26))
        self.label_30 = QLabel(self.tab_metrics)
        self.label_30.setObjectName(u"label_30")
        self.label_30.setGeometry(QRect(570, 270, 111, 16))
        self.tabWidget.addTab(self.tab_metrics, "")
        self.tab_tranfer = QWidget()
        self.tab_tranfer.setObjectName(u"tab_tranfer")
        self.gv_transf_out = QGraphicsView(self.tab_tranfer)
        self.gv_transf_out.setObjectName(u"gv_transf_out")
        self.gv_transf_out.setEnabled(True)
        self.gv_transf_out.setGeometry(QRect(160, 70, 371, 331))
        self.gv_transf_out.setAutoFillBackground(True)
        self.label_4 = QLabel(self.tab_tranfer)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setGeometry(QRect(160, 50, 191, 16))
        self.label_4.setFont(font2)
        self.pb_transfer = QPushButton(self.tab_tranfer)
        self.pb_transfer.setObjectName(u"pb_transfer")
        self.pb_transfer.setGeometry(QRect(560, 180, 151, 32))
        self.pb_reset_transfer = QPushButton(self.tab_tranfer)
        self.pb_reset_transfer.setObjectName(u"pb_reset_transfer")
        self.pb_reset_transfer.setGeometry(QRect(560, 210, 151, 32))
        self.pb_export_transfer = QPushButton(self.tab_tranfer)
        self.pb_export_transfer.setObjectName(u"pb_export_transfer")
        self.pb_export_transfer.setGeometry(QRect(560, 240, 151, 32))
        self.prog_epoch = QProgressBar(self.tab_tranfer)
        self.prog_epoch.setObjectName(u"prog_epoch")
        self.prog_epoch.setGeometry(QRect(160, 450, 541, 43))
        self.prog_epoch.setMaximum(9)
        self.prog_epoch.setValue(0)
        self.prog_epoch.setInvertedAppearance(False)
        self.label_5 = QLabel(self.tab_tranfer)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setGeometry(QRect(160, 440, 91, 16))
        self.la_epoch = QLabel(self.tab_tranfer)
        self.la_epoch.setObjectName(u"la_epoch")
        self.la_epoch.setGeometry(QRect(260, 440, 60, 16))
        self.la_progress = QLabel(self.tab_tranfer)
        self.la_progress.setObjectName(u"la_progress")
        self.la_progress.setGeometry(QRect(260, 440, 161, 16))
        self.tabWidget.addTab(self.tab_tranfer, "")

        self.horizontalLayout_3.addWidget(self.tabWidget)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1200, 24))
        self.menu_File = QMenu(self.menubar)
        self.menu_File.setObjectName(u"menu_File")
        self.menuView = QMenu(self.menubar)
        self.menuView.setObjectName(u"menuView")
        MainWindow.setMenuBar(self.menubar)

        self.menubar.addAction(self.menu_File.menuAction())
        self.menubar.addAction(self.menuView.menuAction())
        self.menu_File.addAction(self.actionExit)
        self.menuView.addAction(self.actionHelp)
        self.menuView.addAction(self.actionAbout)

        self.retranslateUi(MainWindow)

        self.tabWidget.setCurrentIndex(0)
        self.tab_select_metric.setCurrentIndex(4)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"ColorI - DT", None))
        self.actionExit.setText(QCoreApplication.translate("MainWindow", u"Exit", None))
        self.actionHelp.setText(QCoreApplication.translate("MainWindow", u"Help", None))
        self.actionAbout.setText(QCoreApplication.translate("MainWindow", u"Credits", None))
        self.pb_ref_select.setText(QCoreApplication.translate("MainWindow", u"Select", None))
        self.pb_ref_reset.setText(QCoreApplication.translate("MainWindow", u"Reset", None))
        self.pb_test_select.setText(QCoreApplication.translate("MainWindow", u"Select", None))
        self.pb_test_reset.setText(QCoreApplication.translate("MainWindow", u"Reset", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"Reference Image", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"Test Image", None))
        self.label_26.setText(QCoreApplication.translate("MainWindow", u"Height:", None))
        self.label_27.setText(QCoreApplication.translate("MainWindow", u"Width:", None))
        self.label_28.setText(QCoreApplication.translate("MainWindow", u"Height:", None))
        self.label_29.setText(QCoreApplication.translate("MainWindow", u"Width:", None))
        self.la_test_h.setText("")
        self.la_test_w.setText("")
        self.la_ref_h.setText("")
        self.la_ref_w.setText("")
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"Color Difference Output", None))
        self.pb_diff_calc.setText(QCoreApplication.translate("MainWindow", u"Measure difference", None))
        self.pb_diff_reset.setText(QCoreApplication.translate("MainWindow", u"Reset", None))
        self.pb_export_diff.setText(QCoreApplication.translate("MainWindow", u"Export to .tiff", None))
        self.label_6.setText(QCoreApplication.translate("MainWindow", u"Parameters:", None))
        self.label_14.setText(QCoreApplication.translate("MainWindow", u"No editable parameters for this metric", None))
        self.pb_about_rgb.setText(QCoreApplication.translate("MainWindow", u"About this metric", None))
        self.tab_select_metric.setTabText(self.tab_select_metric.indexOf(self.tab_dergb), QCoreApplication.translate("MainWindow", u"DE RGB", None))
        self.label_7.setText(QCoreApplication.translate("MainWindow", u"Parameters:", None))
        self.label_15.setText(QCoreApplication.translate("MainWindow", u"No editable parameters for this metric", None))
        self.pb_about_76_lab.setText(QCoreApplication.translate("MainWindow", u"About this metric", None))
        self.tab_select_metric.setTabText(self.tab_select_metric.indexOf(self.tab_cie76lab), QCoreApplication.translate("MainWindow", u"DE76 (L*a*b*)", None))
        self.label_8.setText(QCoreApplication.translate("MainWindow", u"Parameters:", None))
        self.label_18.setText(QCoreApplication.translate("MainWindow", u"KL", None))
        self.label_19.setText(QCoreApplication.translate("MainWindow", u"K1", None))
        self.label_20.setText(QCoreApplication.translate("MainWindow", u"K2", None))
        self.pb_about_94_lab.setText(QCoreApplication.translate("MainWindow", u"About this metric", None))
        self.tab_select_metric.setTabText(self.tab_select_metric.indexOf(self.tab_cie94lab), QCoreApplication.translate("MainWindow", u"DE94 (L*a*b*)", None))
        self.label_9.setText(QCoreApplication.translate("MainWindow", u"Parameters:", None))
        self.label_21.setText(QCoreApplication.translate("MainWindow", u"KL", None))
        self.label_22.setText(QCoreApplication.translate("MainWindow", u"K1", None))
        self.label_23.setText(QCoreApplication.translate("MainWindow", u"K2", None))
        self.pb_about_00_lab.setText(QCoreApplication.translate("MainWindow", u"About this metric", None))
        self.tab_select_metric.setTabText(self.tab_select_metric.indexOf(self.tab_cie00lab), QCoreApplication.translate("MainWindow", u"DE00(L*a*b*)", None))
        self.label_10.setText(QCoreApplication.translate("MainWindow", u"Parameters:", None))
        self.label_16.setText(QCoreApplication.translate("MainWindow", u"No editable parameters for this metric", None))
        self.pb_about_76_luv.setText(QCoreApplication.translate("MainWindow", u"About this metric", None))
        self.tab_select_metric.setTabText(self.tab_select_metric.indexOf(self.tab_cie76luv), QCoreApplication.translate("MainWindow", u"DE76 (L*u*v*)", None))
        self.label_11.setText(QCoreApplication.translate("MainWindow", u"Parameters:", None))
        self.label_17.setText(QCoreApplication.translate("MainWindow", u"Reference angle \u00b0", None))
        self.pb_about_icsm_luv.setText(QCoreApplication.translate("MainWindow", u"About this metric", None))
        self.tab_select_metric.setTabText(self.tab_select_metric.indexOf(self.tab_icsm), QCoreApplication.translate("MainWindow", u"ICSM22", None))
        self.label_12.setText(QCoreApplication.translate("MainWindow", u"Parameters:", None))
        self.label_13.setText(QCoreApplication.translate("MainWindow", u"Ratio l:c", None))
        self.pb_about_cmc_lab.setText(QCoreApplication.translate("MainWindow", u"About this metric", None))
        self.cb_ratio.setItemText(0, QCoreApplication.translate("MainWindow", u"2:1", None))
        self.cb_ratio.setItemText(1, QCoreApplication.translate("MainWindow", u"1:1", None))

        self.tab_select_metric.setTabText(self.tab_select_metric.indexOf(self.tab_cmc), QCoreApplication.translate("MainWindow", u"DE CMC (L*a*b*)", None))
        self.cb_metrics.setItemText(0, QCoreApplication.translate("MainWindow", u"DE(RGB)", None))
        self.cb_metrics.setItemText(1, QCoreApplication.translate("MainWindow", u"DE76(L*a*b*)", None))
        self.cb_metrics.setItemText(2, QCoreApplication.translate("MainWindow", u"DE94(L*a*b*)", None))
        self.cb_metrics.setItemText(3, QCoreApplication.translate("MainWindow", u"DE00(L*a*b*)", None))
        self.cb_metrics.setItemText(4, QCoreApplication.translate("MainWindow", u"DE76(L*u*v*)", None))
        self.cb_metrics.setItemText(5, QCoreApplication.translate("MainWindow", u"ICSM(L*u*v*)", None))
        self.cb_metrics.setItemText(6, QCoreApplication.translate("MainWindow", u"DECMC(L*a*b*)", None))

        self.label_30.setText(QCoreApplication.translate("MainWindow", u"Selected metric:", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_metrics), QCoreApplication.translate("MainWindow", u"Color Difference Measurement", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"Neural Style Transfer Output", None))
        self.pb_transfer.setText(QCoreApplication.translate("MainWindow", u"Transfer Style", None))
        self.pb_reset_transfer.setText(QCoreApplication.translate("MainWindow", u"Reset ", None))
        self.pb_export_transfer.setText(QCoreApplication.translate("MainWindow", u"Export to .tiff", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"Current Epoch:", None))
        self.la_epoch.setText("")
        self.la_progress.setText("")
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_tranfer), QCoreApplication.translate("MainWindow", u"Neural Style Transfer", None))
        self.menu_File.setTitle(QCoreApplication.translate("MainWindow", u"&File", None))
        self.menuView.setTitle(QCoreApplication.translate("MainWindow", u"View", None))
    # retranslateUi

