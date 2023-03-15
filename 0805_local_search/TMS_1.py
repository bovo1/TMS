from Parser import TMSParser 
from Component import *
import time,sys
from PyQt5.QtWidgets import *
from PyQt5 import uic


form_class = uic.loadUiType("untitled.ui")[0]

class MyWindow(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("TMS")
        self.Btn1.clicked.connect(self.btn_cliked)

    def confirm_condition(self):
        #조건 확인
        global args_list
        args_list = [None] * 30
        if self.quantity_var.value() != None:
            args_list[0] = float(self.quantity_var.value())

        if self.hub_radius_var.value() != None:
            args_list[1] = float(self.hub_radius_var.value())

        if self.crossline_var.value() != None:
            args_list[2] = float(self.crossline_var.value())
        
        if self.car_customer_var.value() != None:
            args_list[14] = float(self.car_customer_var.value())

        args_list[3] = float(self.max_capacity_cost.value())
        args_list[4] = float(self.limit_capacity_var.value())
        args_list[5] = float(self.limit_time_var.value())
        
        if self.max_capacity_cost_var_2.value() != None:
            args_list[6] = float(self.max_capacity_cost_var_2.value())

        if self.timewindow_var.value() != None:
            args_list[7] = float(self.timewindow_var.value())

        if self.car_speed_var.value() != None:
            args_list[8] = float(self.car_speed_var.value())

        args_list[9] = float(self.loading_unloading_var.value())

        if self.no_over_cap.isChecked():
            args_list[12] = True
        else:
            args_list[12] = False
        
        if self.no_over_time.isChecked():
            args_list[13] = True
        else:
            args_list[13] = False
        

    def btn_cliked(self, file_nm):
        QMessageBox.about(self, "확인", self.localBox.currentText())
        self.confirm_condition()
        if self.method1.isChecked():
            #순차
            from Heuristic_copy import TMSSolver
        if self.method2.isChecked():
            #동시
            from Heuristic import TMSSolver
            args_list[10] = float(self.used_limit_time_var.value())
            args_list[11] = float(self.used_limit_cap_var.value())

        file_nm = self.localBox.currentText()
        t = time.time()
        parser = TMSParser(file_nm)
        problem = parser.get_problem()
        solver = TMSSolver(problem)
        ##solver.init_sol_with_districts()

        solver.init_solution(args_list)
        solver.local_search()

        ###p = copy.deepcopy(problem)
        #solver.global_search()
        ####solver.all_local_search()
        painter = Painter()
        painter.draw(problem,file_nm)
        problem.evaluate(t)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = MyWindow()
    myWindow.show()
    app.exec_()
    """global optimize를 Genetic Algorithm을 통해서 구현하기."""
    #file_nm = input('Files : 1110중앙, 1120청주, 1130구미, 1140광주, 1141익산, 1142여수, 1150창원, 1151울산, 1160파주, 1190미국 ')
    



