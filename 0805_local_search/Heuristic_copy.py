from re import I, U
from turtle import position
from click import pass_context
from matplotlib.pyplot import box
from regex import B
from Component import *
import time, itertools
import numpy as np
class TMSSolver:
    def __init__(self, problem : Problem):
        self.problem = problem

    def init_sol_with_districts(self):
        problem = self.problem
        customers = problem.abst_c 
        for c in customers:
            print('aa :', c.address)
        whole_cars = problem.cars
        car_dict = problem.bind_cars()
        #car_dict = {'중구' : [600-09] ...}
        for district in car_dict:
            car_and_rate = car_dict[district]
            cars = [i[0] for i in car_and_rate]
            rates = [i[1] for i in car_and_rate]
            district_in_cust = district.split()[1]
            for cust in customers:
                if district_in_cust in cust.address:
                    car = np.random.choice(cars, 1, False, list(map(lambda x : x / 100, rates)))[0]
                    rest = [c for c in cars if c != car]
                    if car.cbm + cust.cbm < car.cap:
                        print('load')
                        car.customers.append(cust)
                        cust.loaded = True
                    else:
                        print('full')
                        for i in rest:
                            if i.cbm + cust.cbm < i.cap:
                                print('load2')
                                i.customers.append(cust)
                                cust.loaded = True 
                                break 
                        if cust.loaded != True:
                            print("all full : can't load")

        unloaded = [i for i in customers if i.loaded != True]
        #print(unloaded)

        for car in whole_cars:
            unloaded = [i for i in customers if i.loaded != True]
            for c in unloaded:
                if car.cbm + c.cbm < car.cap:
                    c.loaded = True 
                    car.customers += [c] 
        left = [i for i in unloaded if i.loaded == False] 
        print('init sol done, left customers : ', left)
        total = 0
        for car in whole_cars:
            print(f'{car}', car.customers)
            total += len(car.customers)

        print(f'total customers in cars : {total}, left customers : {len(left)}, total_ given_customers : {len(customers)}')
        return 

    def init_solution(self, args):
        problem = self.problem 
        customers = problem.abst_c
        cars = problem.cars 
        hub = problem.hub
        unloaded = [i for i in customers if i.loaded == False]
        #COST 목록
        #cost_distance = []  #차가 가는 거리 COST 이므로 self.cost라는 속성이 있다.
        #cost_distance_var = 3    #곱해줄 값(중요도)
        cost_quantity = []
        if args[0] == 0:
            cost_quantity_var = 1
        else:
            cost_quantity_var = args[0]

        cost_hub_radius = [] 
        if args[1] == 0:
            cost_hub_radius_var = 2
        else:
            cost_hub_radius_var = args[1]

        if args[14] == 0:
            cost_car_customer_var = 2
        else:
            cost_car_customer_var = args[14]

        #cost_new_car_var = 2

        if args[2] == 0:
            crossline_var = 1000 #차의 속성에 교차를 count하여 그 수만큼 곱해줌.
        else:
            crossline_var = args[2]

        al_cross_var = 100

        if args[3] == 0.0:
            max_capacity_cost = 0.8 #80%
        else:
            max_capacity_cost = args[3]

        if args[4] == 0:
            limit_capacity = 2
        else:
            limit_capacity = args[4]

        if args[5] == 0:
            limit_time = 2
        else:
            limit_time = args[5]

        if args[6] == 0:
            max_capacity_cost_var = 100 #최대중량 어길때마다 100씩 코스트 증가하도록
                    #time window의 경우 직선거리를 기준으로 시간을 계산하도록 한다.  분 당 계산 (시간 당 계산도 가능함.)
                    #h 로 계산하되 코스트를 부여하는 가중치는 분으로 계산하겠다는 뜻. late_time (초과한 시간) 이 예를 들어 0.1h 로 나올 수 있음.
        else:
            max_capacity_cost_var = args[6]
        
        if args[7] == 0:
            timewindow_cost_var = 1000   #시간을 어긴 만큼 증가하는 코스트 양
        else:
            timewindow_cost_var = args[7]

        no_over_time = args[13]

        no_over_capacity = args[12]
        
        if args[8] == 0:
            car_speed = 70      #km/h
        else:
            car_speed = args[8]
        launch_time = 1     #나중에 고려될 수 있는 점심 시간. 1은 시간 기준. 
        
        if args[9] == 0:
            loading_unloading_time = 2  #상하차 시간 개당 2분? 정도로...
        else:
            loading_unloading_time = args[9]      

        #물품을 차에 넣어주는 함수 (중복되는 구간이 많으므로 함수를 만들었다.)
        def assign(car_idx, item_idx):
            if sorted_total_cost2[item_idx][0].loaded == False:
                cars[car_idx].dist_cost += cars[car_idx].route[-2].distanceTo(sorted_total_cost2[item_idx][0])# * cost_distance_var
                cars[car_idx].travel_time += haversine(cars[car_idx].route[-2].position, sorted_total_cost2[item_idx][0].position, unit='km') / car_speed
                cars[car_idx].total_cost += sorted_total_cost2[item_idx][1]
                sorted_total_cost2[item_idx][0].loaded = True
                cars[car_idx].customers += [sorted_total_cost2[item_idx][0]]
                cars[car_idx].travel_time += (loading_unloading_time / 60) * len(sorted_total_cost2[item_idx][0].customers_list)
                cars[car_idx].real_cbm += sorted_total_cost2[item_idx][0].cbm
                cars[car_idx].cross_count += sorted_total_cost2[item_idx][2]
                print(cars[car_idx], "에 ", sorted_total_cost2[item_idx][0], "을 적재.")
                unloaded.remove(sorted_total_cost2[item_idx][0])
                
        #시간 조건
        def time_condition(car_idx, item_idx,  car_times, car_speed=car_speed):
            if sorted_total_cost2[item_idx][0].loaded == False:
                #이 거리는 경우의 수에 따라 임시적이고 가변적인 값이 될 수 있으므로
                #차의 속성이 아닌 지역 변수와 리스트에 보관해주도록 한다.
                dist = cars[car_idx].route[-2].distanceTo(sorted_total_cost2[item_idx][0])
                dist_time = dist / car_speed   #km/h 기준
                item_loading_time = (loading_unloading_time / 60 )* len(sorted_total_cost2[item_idx][0].customers_list)
                total_time = dist_time + item_loading_time
                car_times = [cars[car_idx], sorted_total_cost2[item_idx][0], dist, total_time]
            else:
                car_times = None
            return car_times

        def ccw(a, b, c):
            #a -> (1, 2)
            #return 값이 양수면 반시계 방향, 음수면 시계방향, 0 이면 평행
            #return a[0]*b[1] + b[0]*c[1] + c[0]*a[1] - (b[0]*a[1] + c[0]*b[1] + a[0]*c[1])
            result = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])

            if result > 0:
                return 1
            elif result < 0:
                return -1
            else:
                return 0

        #선이 교차하는지 여부
        def intersect_line(a,b,c,d):
            if ccw(a,b,c) * ccw(a,b,d) < 0:
                #교차
                if ccw(c,d,a) * ccw(c,d,b) < 0:
                    return True
            else:
                return False


        #사전에 cost를 다 정리하도록 함.
        dist_customer_hub = [haversine(hub.position, i.position) for i in customers]
        hub_radius = sum(dist_customer_hub) / len(dist_customer_hub)
        
        total_cost = []
        for i in range(len(unloaded)):
            cost_quantity.append((unloaded[i], unloaded[i].cbm * cost_quantity_var))
            if dist_customer_hub[i] > hub_radius:
                cost_hub_radius.append((unloaded[i], dist_customer_hub[i]  * cost_hub_radius_var))
            else:
                cost_hub_radius.append((unloaded[i], 0))
            total_cost.append((unloaded[i], cost_hub_radius[i][1] + cost_quantity[i][1]))
        tc = 0
        tq = 0
        for i in range(len(total_cost)):
            tc += total_cost[i][1]
            tq += unloaded[i].cbm

        #총 몇 대의 차량을 써야하는지 계산함. -> cost 제한과 분배를 위함임.
        predict_car_num = []
        new_predict = []
  
        #조합을 사용함. 정규차량 중에서 r개를 뽑아서 더했을 때 물품을 다 넣을 수 있는지
        #그중에서도 가장 작은 r + 예비 차량
        #일단 정규차량만 쓰도록 함.
        regular_car = []
        for i in range(len(cars)):
            if cars[i].hired_car != 'X':
                regular_car.append(cars[i])
        #print(regular_car)
        for i in range(len(regular_car)):
            predict_car_num.append(list(itertools.combinations((regular_car), i+1)))

        for i in range(len(predict_car_num)):

            for j in range(len(predict_car_num[i])):
                total_car_cap = 0
                for k in range(len(predict_car_num[i][j])):
                    total_car_cap += predict_car_num[i][j][k].cap
                if total_car_cap >= tq:
                    new_predict.append(predict_car_num[i][j])

        #여러 조합이 들어올텐데 그중에서도 가장 짧은 길이를 가진 것을 고름.
        #어차피 1개의 조합부터 차례대로 들어온 것이므로 [0]을 고르면 된다.
        #print(new_predict)
        
        #수량 코스트와 허브 반경 코스트는 루프 내에서 변하지 않는 것이다.
        #근데 아직 최소로 필요한 차의 개수에 맞춰 물품을 효율적으로 넣는 단계는 아니므로
        #나중에 적용시키도록 함.
    
        #cars = new_predict[0]

        #이제 루프를 돌면서 물품을 넣을 차례
        car_idx = 0
        item_idx = 0
        while len(unloaded) > 0:
            while car_idx < len(cars):
                total_cost2 = []
                for i in unloaded:
                    for j in range(len(total_cost)):
                        if total_cost[j][0] == i:
                            total_cost2.append([i, cars[car_idx].dist_cost + total_cost[j][1], 0])  
                            #total_cost2의 마지막은 교차 횟수를 넣기 위함이다.  
                sorted_dist = []
                for i in range(len(total_cost2)):
                    if len(cars[car_idx].route) >= 3: 
                        sorted_dist.append((i, haversine(cars[car_idx].route[-2].position, unloaded[i].position)))
                        total_cost2[i][1] +=  (haversine(cars[car_idx].route[-2].position, unloaded[i].position) * cost_car_customer_var)
                
                sorted_dist = sorted(sorted_dist, key=lambda x: x[1])

                for i in range(len(sorted_dist)):
                    for j in range(len(total_cost2)):
                        if sorted_dist[i][0] == total_cost2[j][0]:
                            total_cost2[j][1] += sorted_dist[i][1] * i * cost_car_customer_var


                #현재 차량의 위치에 맞춰 통합 코스트에 현재 차량 위치와 남은 거점들간의 거리를 추가함.
                sorted_total_cost2 = sorted(total_cost2, key=lambda x: x[1], reverse=False)
                #최대중량 조건 체크
                #total_cost2의 구조 : 아이템, 코스트(차의 거리 코스트 + 기본 코스트)
                for k in range(len(sorted_total_cost2)):
                    #print("물품:",sorted_total_cost2[k][0])
                    if cars[car_idx].real_cbm + sorted_total_cost2[k][0].cbm <= cars[car_idx].cap:
                        if cars[car_idx].real_cbm + sorted_total_cost2[k][0].cbm >= cars[car_idx].cap * max_capacity_cost:
                            #최대 중량을 어기므로 추가 코스트
                            over_capacity = ((cars[car_idx].real_cbm + sorted_total_cost2[k][0].cbm) - (cars[car_idx].cap * max_capacity_cost))
                            print("최대 중량이", over_capacity,"만큼 초과되어 코스트를 추가함. 추가 전: ", sorted_total_cost2[k][1], "추가 후: ",sorted_total_cost2[k][1] + (max_capacity_cost_var * over_capacity) )
                            sorted_total_cost2[k][1] += max_capacity_cost_var * over_capacity

                        #교차 검사
                        cars[car_idx].item_cross_count = 0
                        for rc in range(len(cars)):
                            if len(cars[rc].route) >= 3:
                                for rc2 in range(len(cars[rc].route)-2):
                                    if intersect_line(cars[rc].route[rc2].position, cars[rc].route[rc2+1].position, cars[car_idx].route[-2].position, sorted_total_cost2[k][0].position) == True:
                                        #만약 교차가 있다면 코스트 부여
                                        cars[car_idx].item_cross_count += 1
                                        sorted_total_cost2[k][2] = cars[car_idx].item_cross_count
                                        print("현재 진행중인 차량",cars[car_idx], "에서 갈 후보 물품인", sorted_total_cost2[k][0], "에서", cars[rc],"의 경로",cars[rc].route[rc2], "와 ",cars[rc].route[rc2+1]  ,"에서 교차가 발생하였고 현재", cars[car_idx], "의 교차 count는 ", cars[car_idx].cross_count, "이며,", sorted_total_cost2[k][0], "에서의 count는", cars[car_idx].item_cross_count," 이므로 코스트를 추가함. 추가 전: ", sorted_total_cost2[k][1], "추가 후: ", sorted_total_cost2[k][1] + (cars[car_idx].item_cross_count * crossline_var) )
                                        sorted_total_cost2[k][1] += cars[car_idx].item_cross_count * crossline_var

                        #시간 초과 검사
                        car_times = []
                        car_time = time_condition(car_idx, k, car_times)
                        if car_time != None:
                            if cars[car_idx].travel_time + car_time[3] >= cars[car_idx].whole_travel_time:
                                late_time = (cars[car_idx].travel_time + car_time[3]) - cars[car_idx].whole_travel_time
                                late_time = late_time * 60
                                print("근무 시간이 ", late_time, "만큼 초과하여 코스트를 부과함. 추가 전: ", sorted_total_cost2[k][1], "추가 후: ", sorted_total_cost2[k][1] + (late_time * timewindow_cost_var))
                                sorted_total_cost2[k][1] += late_time * timewindow_cost_var

                sorted_total_cost2 = sorted(sorted_total_cost2, key=lambda x: x[1])
                #정렬된 코스트를 바탕으로 시간과 용량 조건에 따라 넣을 수 있는 물품을 넣는다.
                sorted_total_cost2_time_condition = []
                available_car_item_list = []
                car_times = []
                for stc in range(len(sorted_total_cost2)):
                    sorted_total_cost2_time_condition.append(time_condition(car_idx, stc, car_times))

                for stctc in sorted_total_cost2_time_condition:
                    if no_over_capacity == True:
                        if stctc[0].real_cbm + stctc[1].cbm <= stctc[0].cap * max_capacity_cost:
                            if no_over_time == True:
                                if stctc[0].travel_time + stctc[3] <= stctc[0].whole_travel_time:
                                    available_car_item_list.append(stctc[1])
                            else:
                                if stctc[0].travel_time + stctc[3] <= stctc[0].whole_travel_time + limit_time:
                                    available_car_item_list.append(stctc[1])
                    else:
                        #과적O
                        if stctc[0].real_cbm + stctc[1].cbm <= stctc[0].cap - limit_capacity:
                            if no_over_time == True:
                                #초과근무X
                                if stctc[0].travel_time + stctc[3] <= stctc[0].whole_travel_time:
                                    available_car_item_list.append(stctc[1])
                            else:
                                #초과근무O
                                if stctc[0].travel_time + stctc[3] <= stctc[0].whole_travel_time + limit_time:
                                    available_car_item_list.append(stctc[1])
 
                if len(available_car_item_list) == 0:
                    print("현재", cars[car_idx], "차량에 적재가 불가능하여 새로운 차량을 쓰도록 함.")
                    car_idx += 1
                else:
                    #assign 하기 전에 sorted_total_cost2 에 맞도록 item_idx를 넣어줘야 하기 때문에 찾도록 함.
                    print("적재 가능", available_car_item_list)
                    for x in range(len(sorted_total_cost2)):
                        if available_car_item_list[0] == sorted_total_cost2[x][0]:
                            assign(car_idx, x)
                            available_car_item_list = []
                            break

                if len(unloaded) == 0:
                    break

        
        #print(total_cost2)

        left = [i for i in unloaded if i.loaded == False]

        return

    def three_opt_swap(self, route, i, j):
        k = len(route)
        if i > 0:
            return route[:i] + route[j:i-1:-1] + route[-1:-k+j:- 1]
        else:
            return route[j:i:-1] + [route[i]] + route[-1:-(k-j):-1]

    def local_search(self):
        def one_search(self, car, hub):
            best_dist = self.problem.total_distance(car.route)
            new_custs = []
            new_dists = []
            for i, j in itertools.combinations(range(len(car.customers)), 2):
                new_cust = self.three_opt_swap(car.customers, i, j)
                new_custs.append(new_cust)
                new_dists.append(self.problem.total_distance([hub, *new_cust, hub]))
            new_routes = list(zip(new_custs, new_dists)) 
            best_route = min(new_routes, key = lambda x : x[1])
            if best_route[1] < best_dist:
                car.customers = best_route[0]
                best_dist = best_route[1]
                return True 
            return False

        hub = self.problem.hub 
        for car in self.problem.cars:
            if len(car.customers) <= 1:
                continue 
            num = 0
            repeat = True
            while repeat:
                repeat = one_search(self, car, hub)

    def partial_local_search(self, hub, customers):
        if len(customers) <= 1 :
            return customers
        
        length = len(customers)
        saver = customers[:]
        best_dist = self.problem.total_distance([hub, *customers, hub])
        cust = customers.pop() 
        for i in range(len(customers)):
            customers.insert(i, cust) 
            if self.problem.total_distance([hub, *customers, hub]) < best_dist:
                break 
            else:
                customers.remove(cust)

        print('done')
        if len(customers) < length:
            customers.append(cust)
        print('before', saver)
        print('after', customers)

        return customers
        

    def all_local_search(self):
        """used to find the best absolute solution. works only for 광주"""
        hub = self.problem.hub 
        for car in self.problem.cars:
            if len(car.customers) == 0:
                continue
            

            best_customers = car.customers
            best_distance = self.problem.total_distance([hub, *best_customers, hub])
            permutation = itertools.permutations(best_customers)
            all_routes = list(permutation)

            all_distances = [[self.problem.total_distance([hub,*i,hub]), idx] for idx,i in enumerate(all_routes)]
            best_distance = min(all_distances, key = lambda x : x[0])
            print(list(all_routes[best_distance[1]]))
            car.customers = list(all_routes[best_distance[1]])
        print('----------------------------------------------------------')
        return 


    #def send(self, c1 : list[Customer] , c2, i):
    def send(self, c1 : list , c2, i):
        if len(c1) == 0:
            return c1[:], c2[:]
        c1 = c1[:]
        c2 = c2[:]
        cust = c1.pop(i) 
        c2.append(cust)

        return c1, c2

    #def swap(self, c1 : list[Customer], c2 : list[Customer], i , j):
    def swap(self, c1 : list, c2 : list, i , j):
        c1 = c1[:]
        c2 = c2[:]
        c1[i], c2[j] = c2[j], c1[i] 
        return c1, c2

    def global_search(self):
        t1 = time.time()
        def sending_one_time(self, car1, car2, hub):
            for a in range(len(car1.customers)):
                
                if car2.cbm + car1.customers[a].cbm > car2.cap:
                    continue 

                cust1, cust2 = self.send(car1.customers, car2.customers, a) 
                r1, r2 = map((lambda x : [hub, *x, hub]), [cust1, cust2])
                if self.problem.objective_function2([r1, r2]) < self.problem.objective_function2([car1.route, car2.route]):
                    car1.customers, car2.customers = cust1, cust2 
                    return True
            return False

        def swapping_one_time(self, car1, car2, hub):
            for i in range(len(car1.customers)):
                for j in range(len(car2.customers)):
                    if car1.cbm + (car2.customers[j].cbm - car1.customers[i].cbm) > car2.cap and car2.cbm + (car1.customers[i].cbm - car2.customers[j].cbm) > car1.cap:
                        #원본
                        #cust1, cust2 = self.swap(c1, c2, i, j)
                        #수정
                        cust1, cust2 = self.swap(car1.customers, car2.customers, i, j)
                        r1, r2 = map((lambda x : [hub, *x, hub], [cust1, cust2]))
                        if self.problem.objective_function2([r1, r2]) < self.problem.objective_function2([car1.route, car2.route]):
                            car1.customers, car2.customers = cust1, cust2 
                            return True 
            return False
        #----------------------------------------------------------------------------------------------------
        hub = self.problem.hub 

        for i in range(len(self.problem.cars)):
            for j in range(i+1, len(self.problem.cars)):
                car1, car2 = self.problem.cars[i], self.problem.cars[j] 
                if len(car1.customers) + len(car2.customers) <= 2:
                    continue

                while sending_one_time(self, car1, car2, hub):
                    if time.time() - t1 > 10:
                        print('working...')
                        t1 = time.time()

                while swapping_one_time(self, car1, car2, hub):
                    if time.time() - t1 > 10:
                        print('working...')
                        t1 = time.time
        self.local_search()