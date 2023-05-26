# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'no_image_dialog.ui'
##
## Created by: Qt User Interface Compiler version 6.4.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QDialog, QLabel, QPushButton,
    QSizePolicy, QWidget)

class Ui_no_image_dialog(object):
    def setupUi(self, no_image_dialog):
        if not no_image_dialog.objectName():
            no_image_dialog.setObjectName(u"no_image_dialog")
        no_image_dialog.resize(350, 80)
        sizePolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(no_image_dialog.sizePolicy().hasHeightForWidth())
        no_image_dialog.setSizePolicy(sizePolicy)
        no_image_dialog.setMinimumSize(QSize(350, 80))
        no_image_dialog.setMaximumSize(QSize(350, 80))
        self.label = QLabel(no_image_dialog)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(120, 10, 121, 21))
        self.label.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignTop)
        self.label.setWordWrap(False)
        self.pb_cancel = QPushButton(no_image_dialog)
        self.pb_cancel.setObjectName(u"pb_cancel")
        self.pb_cancel.setGeometry(QRect(120, 40, 113, 32))
        sizePolicy.setHeightForWidth(self.pb_cancel.sizePolicy().hasHeightForWidth())
        self.pb_cancel.setSizePolicy(sizePolicy)

        self.retranslateUi(no_image_dialog)

        QMetaObject.connectSlotsByName(no_image_dialog)
    # setupUi

    def retranslateUi(self, no_image_dialog):
        no_image_dialog.setWindowTitle(QCoreApplication.translate("no_image_dialog", u"Warning", None))
        self.label.setText(QCoreApplication.translate("no_image_dialog", u"No image to save! ", None))
        self.pb_cancel.setText(QCoreApplication.translate("no_image_dialog", u"Cancel", None))
    # retranslateUi

