import sys
from PySide6.QtWidgets import QMainWindow, QApplication, QDialog, QFileDialog, QGraphicsScene, QGraphicsPixmapItem, QSplashScreen 
from PySide6.QtGui import QImage, QPixmap, QIcon
from PySide6.QtCore import QObject, QThread, QTimer, QCoreApplication
from PIL import Image
import numpy as np
import os
import subprocess
from time import sleep
from gui.gui import Ui_MainWindow
from gui.shape_error_dialog import Ui_Dialog as Ui_Dialog_shape_error
from gui.about_dialog import Ui_Dialog as Ui_Dialog_about
from gui.no_image_dialog import Ui_no_image_dialog as Ui_no_image_dialog
from utils.neural_style_transfer import StyleTransfer
from utils.color_difference_metrics import ColorDifferenceMetrics


class Gui(QMainWindow, Ui_MainWindow, Ui_Dialog_shape_error, Ui_Dialog_about, Ui_no_image_dialog, QObject):
    def __init__(self):
        super(Gui, self).__init__()
        self.setupUi(self)
        self.show()
        self.nst = StyleTransfer()
        self.cdm = ColorDifferenceMetrics()
        self.selected_metric = self.cb_metrics.currentText()
        self.reference_image_path = None
        self.test_image_path = None
        self.transf_output_image = None
        self.diff_output_image = None
        self.test_image = None
        self.reference_image = None
        self.pb_ref_select.clicked.connect(self.select_image)
        self.pb_test_select.clicked.connect(self.select_image)
        self.pb_ref_reset.clicked.connect(self.reset_reference_image)
        self.pb_test_reset.clicked.connect(self.reset_test_image)
        self.pb_diff_calc.clicked.connect(self.execute_color_difference)
        self.pb_transfer.clicked.connect(self.execute_style_transfer)
        self.pb_diff_reset.clicked.connect(self.reset_diff_image)
        self.pb_reset_transfer.clicked.connect(self.reset_transf_image)
        self.pb_export_diff.clicked.connect(self.save_image)
        self.pb_export_transfer.clicked.connect(self.save_image)
        self.actionAbout.triggered.connect(self.open_about)
        self.actionExit.triggered.connect(self.close)
        self.prog_epoch.setMaximum(self.nst.epochs-1)
        self.pb_transfer.setEnabled(False)
        self.pb_diff_calc.setEnabled(False)
        self.actionHelp.triggered.connect(self.open_pdf)
        self.pb_about_94_lab.clicked.connect(self.open_pdf)
        self.pb_about_00_lab.clicked.connect(self.open_pdf)
        self.pb_about_76_lab.clicked.connect(self.open_pdf)
        self.pb_about_cmc_lab.clicked.connect(self.open_pdf)
        self.pb_about_icsm_luv.clicked.connect(self.open_pdf)
        self.pb_about_76_luv.clicked.connect(self.open_pdf)
        self.pb_about_rgb.clicked.connect(self.open_pdf)
        self.cb_metrics.currentIndexChanged.connect(self.select_metric)
        self.scene = QGraphicsScene()
        self.gv_transf_out.setScene(self.scene)
        #self.transfer_running = False
        self.transfer_canceled = False

        # Connect signals and slots for the neural style transfer
        #self.nst_thread = QThread()
  


    ## Gui function to open about dialog
    def open_about(self):
        self.about_dialog = QDialog()
        self.dialog_ui = Ui_Dialog_about()
        self.dialog_ui.setupUi(self.about_dialog)
        self.about_dialog.show()

    def select_image(self, image_path):
        self.select_dialog = QFileDialog()
        self.select_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        self.select_dialog.setNameFilter('Images (*.jpg *.jpeg *.tiff *.tif)')
        self.select_dialog.show()
        if self.select_dialog.exec():
            image_path = self.select_dialog.selectedFiles()[0]
        else:
            image_path = None
        image = Image.open(image_path)
        if image_path.endswith('.tiff'):
                image = image.convert('RGB')
            # Convert image to numpy array ans set data type to uint8 
        image = np.array(image)
        image = image.astype('uint8')
        # Show image in the gui
        scene = QGraphicsScene()
        qimage = QImage(image, image.shape[1], image.shape[0], QImage.Format.Format_RGB888)
        # Resize image to fit in the graphics view
        qimage = qimage.scaled(191, 181)
        pixmap = QPixmap.fromImage(qimage)
        scene.addPixmap(pixmap)
        # Get the sender object (the button that was clicked)
        sender = self.sender()
        # Check which button was clicked and set the corresponding image
        if sender == self.pb_ref_select:
            self.reference_image = image
            self.gv_ref_img.setScene(scene)
            # Set the image size labels
            self.la_ref_h.setText(str(image.shape[0]) + 'px')
            self.la_ref_w.setText(str(image.shape[1]) + 'px')
        elif sender == self.pb_test_select:
            self.test_image = image
            self.gv_test_img.setScene(scene)
            # Set the image size labels
            self.la_test_h.setText(str(image.shape[0]) + 'px')
            self.la_test_w.setText(str(image.shape[1])+ 'px')
        # Enable the transfer button if both images are selected
        if np.all(self.reference_image != None) and np.all(self.test_image != None):
            self.pb_transfer.setEnabled(True)
            self.pb_diff_calc.setEnabled(True)
        else:
            self.pb_transfer.setEnabled(False)
            self.pb_diff_calc.setEnabled(False)
        
        
    def save_image(self):
        # Get the sender object (the button that was clicked)

        sender = self.sender()
        # Check which button was clicked and set the corresponding image

        if sender == self.pb_export_diff:
            out_image = self.diff_output_image

        elif sender == self.pb_export_transfer:
            out_image = self.transf_output_image    

        else:
            out_image = None
        
        if out_image is None:
            # If the output image is None, pop up a warning dialog
            self.no_image_dialog = QDialog()
            self.no_image_ui = Ui_no_image_dialog()
            self.no_image_ui.setupUi(self.no_image_dialog)
            self.no_image_dialog.show()
            self.no_image_ui.pb_cancel.clicked.connect(self.no_image_dialog.close)
        else:
            #pixmap = QPixmap.fromImage(self.diff_output_image)
            image =Image.fromarray(out_image)
            filename, _ = QFileDialog.getSaveFileName(self, 'Save File', '', 'Images (*.tiff)')
            if filename != '':
                image.save(filename, 'TIFF')
                #imageio.imwrite(filename, out_image, format='tiff')

    def reset_reference_image(self):
        self.reference_image_path = None
        self.reference_image = None
        self.gv_ref_img.setScene(None)
        self.enable_transfer()
        self.enable_diff_calc()
        self.la_ref_h.setText('')
        self.la_ref_w.setText('')

    def reset_test_image(self):
        self.test_image_path = None
        self.test_image = None
        self.gv_test_img.setScene(None)
        self.enable_transfer()
        self.enable_diff_calc()
        self.la_test_h.setText('')
        self.la_test_w.setText('')

    def reset_transf_image(self):
        self.output_transf_image = None
        self.gv_transf_out.setScene(None)
        self.prog_epoch.setValue(0)
        self.la_progress.setText('')
        self.cancel_transfer()


    def reset_diff_image(self):
        self.output_diff_image = None
        self.gv_metrics_out.setScene(None)
    
    def check_shape(self,reference_image, test_image):
        # Control if the images match in size
        #shape_dialog = Ui_Dialog_shape_error()
        if reference_image.shape != test_image.shape:
            self.shape_error_dialog = QDialog()
            self.shape_error_ui = Ui_Dialog_shape_error()
            self.shape_error_ui.setupUi(self.shape_error_dialog)
            self.shape_error_dialog.show()
            self.shape_error_ui.pushButton.clicked.connect(self.shape_error_dialog.close)
            return False
        else:
            return True
    
    def open_pdf(self):
        # Open tjhe documentation pdf according to the OS
        if sys.platform.startswith('darwin'):
            subprocess.call(('open', 'gui/resources/Colori-DT.pdf'))
        elif sys.platform.startswith('cygwin'):
            os.startfile('gui/resources/Colori-DT.pdf')
        elif os.name == 'posix':
            subprocess.call(('xdg-open', 'gui/resources/Colori-DT.pdf'))

    def enable_transfer(self):
        if np.all(self.reference_image != None) and np.all(self.test_image != None):
            self.pb_transfer.setEnabled(True)
        else:
            self.pb_transfer.setEnabled(False)
            
    def enable_diff_calc(self):
        if np.all(self.reference_image != None) and np.all(self.test_image != None):            
            self.pb_diff_calc.setEnabled(True)
        else:
            self.pb_diff_calc.setEnabled(False)

    # Set the selected metric with parameters
    # not collapsed in a single function for future scalability
    def calc_00lab(self,reference_image, test_image):
        k1 = self.sb_k1_00.value()
        k2 = self.sb_k2_00.value()
        kL = self.sb_kL_00.value()
        result = self.cdm.ciede2000(reference_image, test_image, k1, k2, kL)
        return result
    
    def calc_76lab(self, reference_image, test_image):
        # No parameters to set
        result = self.cdm.cie76_lab(reference_image, test_image)
        return result

    def calc_76luv(self, reference_image, test_image):
        # No parameters to set
        result = self.cdm.cie76_luv(reference_image, test_image)
        return result

    def calc_94lab(self, reference_image, test_image):
        k1 = self.sb_k1_94.value()
        k2 = self.sb_k2_94.value()
        kL = self.sb_kL_94.value()
        result = self.cdm.cie94(reference_image, test_image, k1, k2, kL)
        return result
    
    def calc_cmc(self, reference_image, test_image):
        ratio = 1
        # Get the ratio from the gui
        if self.cb_ratio.currentText().startswith('2'):
            ratio = 2
        
        result = self.cdm.cmc(reference_image, test_image, ratio)
        return result

    def calc_icsm(self, reference_image, test_image):
        ref = self.sb_ang.value()
        result = self.cdm.icsm(reference_image, test_image, ref)
        return result

    def calc_rgb(self, reference_image, test_image):
        result = self.cdm.dergb(reference_image, test_image)
        return result

    def select_metric(self):
        self.selected_metric = self.cb_metrics.currentText()

    def execute_color_difference(self):
        if self.pb_diff_calc.isEnabled:
            if self.check_shape(self.reference_image, self.test_image):
                self.get_color_difference(self.reference_image, self.test_image)
            

    def get_color_difference(self, reference_image, test_image):
        # Define a dictionary with the metrics and the corresponding functions
        metrics = {
            'DE00(L*a*b*)': self.calc_00lab,
            'DE76(L*a*b*)': self.calc_76lab,
            'DE76(L*u*v*)': self.calc_76luv,
            'DE94(L*a*b*)': self.calc_94lab,
            'DECMC(L*a*b*)': self.calc_cmc,
            'ICSM(L*u*v*)': self.calc_icsm,
            'DE(RGB)': self.calc_rgb
        }
            # Get the selected metric and execute the corresponding function
        result = metrics[self.selected_metric](reference_image, test_image)
        self.diff_output_image = result
        # Show the result in the GUI
        scene = QGraphicsScene()
        qimage = QImage(result, result.shape[1], result.shape[0], QImage.Format.Format_Grayscale8)
        # Resize image to fit in the graphics view
        qimage = qimage.scaled(371, 331)
        pixmap = QPixmap.fromImage(qimage)
        scene.addPixmap(pixmap)
        self.gv_metrics_out.setScene(scene)

    ## Neural style transfer
    # Assign the transfer to a separate thread to avoid blocking the GUI and to show progresses while running

    def execute_style_transfer(self):
        # Disable the transfer buttons
        self.transfer_canceled = False
        self.pb_transfer.setEnabled(False)
        self.pb_reset_transfer.setEnabled(False)
        self.pb_export_transfer.setEnabled(False)

        self.nst_thread = QThread()

        # Move the NST object to the thread
        self.nst.moveToThread(self.nst_thread)

        # Connect signals and slots
        self.nst.finished.connect(self.on_transfer_finished)
        self.nst.finished.connect(self.nst.deleteLater)
        self.nst_thread.started.connect(lambda: self.nst.run(self.reference_image, self.test_image))
        self.nst.result.connect(self.update_transf_image)
        self.nst.pb_progress.connect(self.update_progress)

        # Start the thread
        self.nst_thread.start()
    
    def update_transf_image(self, bytearr):
        image_from_bytes = np.frombuffer(bytearr.data(), dtype=np.uint8).reshape(224,224,3)
        # resize image to graphics view size for export

        self.transf_output_image = image_from_bytes
        qimage = QImage(image_from_bytes, image_from_bytes.shape[1], image_from_bytes.shape[0], QImage.Format.Format_RGB888)
        # resize image to fit in the graphics view
        qimage = qimage.scaled(371, 331)
        pixmap = QPixmap.fromImage(qimage)
        item = QGraphicsPixmapItem(pixmap)
        self.scene.clear()
        self.scene.addItem(item)
        self.gv_transf_out.setScene(self.scene)
        self.gv_transf_out.viewport().update()
    
    def update_progress(self, progress):
        # Progress range is 0-9
        self.prog_epoch.setValue(0)
        if self.transfer_canceled:
            return
        self.prog_epoch.setValue(progress)
        if progress < self.nst.epochs-1:
            self.la_progress.setText(str(progress+1))
        else:
            self.la_progress.setText('Transfer complete')

        self.prog_epoch.repaint()
        self.prog_epoch.update()


    def on_transfer_finished(self):
        # Disconnect signals and slots
        self.nst.finished.disconnect(self.on_transfer_finished)
        self.nst.result.disconnect(self.update_transf_image)
        self.nst.pb_progress.disconnect(self.update_progress)

        # Move the NST object back to the main thread
        #self.nst.moveToThread(QCoreApplication.instance().thread())

        # Clean up the thread object
        self.nst_thread.quit()
        self.nst_thread.wait()
        self.nst_thread.deleteLater()

        # Re-enable the transfer buttons
        self.pb_transfer.setEnabled(True)
        self.pb_reset_transfer.setEnabled(True)
        self.pb_export_transfer.setEnabled(True)
    
    def cancel_transfer(self):
        self.transfer_canceled = True
        

def main():
    app = QApplication(['Colori-DT'])
    QCoreApplication.setApplicationName('Colori-DT')
    icon = QIcon('gui/resources/icon.png')
    app.setWindowIcon(icon)
    pixmap = QPixmap('gui/resources/splash.jpg')
    splash = QSplashScreen(pixmap)
    splash.setMask(pixmap.mask())
    splash.show()
    sleep(2)
    app.processEvents()  
    gui = Gui()
    gui.show()
    splash.close()
    sys.exit(app.exec())

