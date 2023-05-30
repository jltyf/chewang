# -*- coding: utf-8 -*-
import os
import sys

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication

from Generator.sc_tool import Work_Model

sys.path.append('../')

from Generator.sc_generation import Task
from MainWindow import Ui_MainWindow


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    # _instance_lock = threading.Lock()

    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.start_button.clicked.connect(self.generate_event)

    # def __new__(cls, *args, **kwargs):
    #     if not hasattr(MainWindow, "_instance"):
    #         with MainWindow._instance_lock:
    #             if not hasattr(MainWindow, "_instance"):
    #                 MainWindow._instance = QtWidgets.QMainWindow.__new__(cls)
    #     return MainWindow._instance

    def generate_event(self):
        self.change_button('disable')
        input_path = self.input_path.toPlainText()
        output_path = self.output_path.toPlainText()
        if not os.path.exists(input_path):
            self.change_text('数据输入目录不存在，请检查！')
            self.change_button('enable')
            return
        if output_path == '':
            output_path = input_path
        elif not os.path.exists(output_path):
            try:
                os.makedirs(output_path)
            except:
                self.change_text('输出目录不存在，且创建失败，请确认！')
                self.change_button('enable')
                return
        model = self.comboBox.currentText()
        if model == '请选择数据模式':
            self.change_text('场景还原需要选择数据格式，请检查或咨询相关人员！')
            self.change_button('enable')
            return
        else:
            self.textBrowser.clear()
            self.generate(model, input_path, output_path)
        self.change_button('enable')

    def change_button(self, style):
        if style == 'disable':
            self.start_button.setEnabled(False)
            self.start_button.setStyleSheet("QPushButton{color:}"
                                            "QPushButton{background-color:rgb(220,220,220)}")
        else:
            self.start_button.setEnabled(True)
            self.start_button.setStyleSheet("QPushButton:hover{background-color:rgb(242,242,242)}"
                                            "QPushButton:pressed{padding-left:3px}"
                                            "QPushButton:pressed{padding-top:3px}")

    def change_text(self, text):
        self.textBrowser.setText(text)
        QApplication.processEvents()

    def generate(self, model, input_path, output_path):
        if model == '路端数据还原':
            work_model = Work_Model.roadside
        elif model == '车端数据还原':
            work_model = Work_Model.car
        else:
            work_model = Work_Model.merge
        new_task = Task(input_path, "data.csv", work_model)
        self.change_text('任务正在运行，请耐心等待')
        # 生成场景
        new_task.batchRun(input_path, output_path, self.textBrowser)


def app_start():
    app = QtWidgets.QApplication(sys.argv)
    longin_window = MainWindow()
    longin_window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    app_start()
