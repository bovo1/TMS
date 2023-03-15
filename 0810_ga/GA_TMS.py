from GA_Heuristic import *
from Parser import *
from PyQt5.QtWidgets import *
from PyQt5 import uic
import sys

form_class = uic.loadUiType("GA.ui")[0]

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
        if self.available_yong.isChecked():
            args_list[23] = True
        else:
            args_list[23] = False
        
        args_list[15] = self.used_limit_time_var.value()
        args_list[16] = self.used_limit_cap_var.value()
        args_list[17] = self.count_var.value()
        args_list[18] = self.yong_car_var.value()
        args_list[19] = self.num_of_generation_var.value()
        args_list[20] = self.no_update_var.value()
        args_list[21] = self.mutant_var.value()
        args_list[22] = self.change_var.value()
        args_list[24] = self.yong_car_select_var.value()
        args_list[25] = self.hub_radius_var_km.value()
        args_list[26] = self.good_gene_deli.value()
        args_list[27] = self.gene_select_per.value()
        args_list[28] = self.more_km_var.value()
        

    def btn_cliked(self, file_nm):
        QMessageBox.about(self, "확인", self.localBox.currentText())
        self.confirm_condition()
        file_nm = self.localBox.currentText()
        parser = TMSParser(file_nm)
        problem = parser.get_problem()
        #solver.init_solution(args_list)
        solver = GA(problem, args_list)
        ga_route = solver.global_search(int(self.count_var.value()), int(self.num_of_generation_var.value()), args_list ,problem)
        customers = problem.abst_c
        painter = Painter(problem)
        painter.position_to_map_coords(customers + [problem.hub])
        painter.draw_points([problem.hub, *problem.abst_c])
        hub_position = problem.customers[0].position
        painter.draw_ga_route(problem.hub, ga_route, hub_position, file_nm, solver.hub_radius)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = MyWindow()
    myWindow.show()
    app.exec_()
