from Parser import TMSParser 
#from Heuristic import TMSSolver
#from Heuristic_copy2 import *
from Heuristic2 import *
from Component import *
import time



"""global optimize를 Genetic Algorithm을 통해서 구현하기."""
#file_nm = input('Files : 1110중앙, 1120청주, 1130구미, 1140광주, 1141익산, 1142여수, 1150창원, 1151울산, 1160파주, 1190미국 ')
file_nm = '1190미국'
t = time.time()
parser = TMSParser(file_nm)

problem = parser.get_problem()
p1 = copy.deepcopy(problem)

solver = TMSSolver(problem)
##solver.init_sol_with_districts()

solver.init_solution()
solver.local_search()

###p = copy.deepcopy(problem)
#solver.global_search()
####solver.all_local_search()
painter = Painter()
painter.draw(problem,file_nm)
problem.evaluate(t)
