from GA_Heuristic import *
from Parser import *
from PyQt5.QtWidgets import *
from PyQt5 import uic

form_class = uic.loadUiType("GA.ui")[0]
#from Heuristic import *
"""GA 리메이크중"""
#file = input('Files : 1110중앙, 1120청주, 1130구미, 1140광주, 1141익산, 1142여수, 1150창원, 1151울산, 1160파주, 1190미국 - ')
file_nm = '1160파주'
parser = TMSParser(file_nm)

problem = parser.get_problem()

solver = GA(problem)
ga_route = solver.global_search(80, 100, problem)
customers = problem.abst_c
#tsp_route = solver.local_search(customers[:30])

painter = Painter(problem)
painter.position_to_map_coords(customers + [problem.hub])
painter.draw_points([problem.hub, *problem.abst_c])
#painter.draw_route([problem.hub, *tsp_route, problem.hub], painter.map)
hub_position = problem.customers[0].position
painter.draw_ga_route(problem.hub, ga_route, hub_position, file_nm)