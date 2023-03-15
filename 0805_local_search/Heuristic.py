from re import I, U
from turtle import position
from click import pass_context
from matplotlib.pyplot import box
from regex import B
from Component import *
from shapely import *
from shapely.ops import split, linemerge
import time, itertools
import numpy as np
import math
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
        #random.shuffle(unloaded)
        #for i in range(len(unloaded)):
            #print("gg!", unloaded[i].customers_list)
        #COST 목록
        #cost_distance = []  #차가 가는 거리 COST 이므로 self.cost라는 속성이 있다.
        #cost_distance_var = 3    #곱해줄 값(중요도)


        #args가 None이 아닐 경우에만 집어넣는다.


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

        al_cross_var = 1500

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


        used_limit_time = args[10]
        used_limit_capacity = args[11]    


        no_over_cap = args[12]
        

        car_speed = args[8]
        launch_time = 1     #나중에 고려될 수 있는 점심 시간. 1은 시간 기준. 
        
        if args[9] == 0:
            loading_unloading_time = 2  #상하차 시간 개당 2분? 정도로...
        else:
            loading_unloading_time = args[9]
        
        print("조건들")
        print(cost_quantity_var, cost_hub_radius_var, crossline_var ,max_capacity_cost, limit_capacity, limit_time, max_capacity_cost_var, timewindow_cost_var, car_speed, loading_unloading_time, args[10], args[11], no_over_cap, no_over_time, cost_car_customer_var)


        #물품을 차에 넣어주는 함수 (중복되는 구간이 많으므로 함수를 만들었다.)
        def assign(car, item, cost):
            if item.loaded == False:
                car.dist_cost += car.route[-2].distanceTo(item)
                car.travel_time += haversine(car.route[-2].position, item.position, unit='km') / car_speed
                car.total_cost += cost
                item.loaded = True
                car.customers += [item]
                car.travel_time += (loading_unloading_time / 60) * len(item.customers_list)
                car.real_cbm += item.cbm
                unloaded.remove(item)

        #시간 조건
        def time_condition(car, item,  car_times, car_speed=car_speed):
            if item.loaded == False:
                #이 거리는 경우의 수에 따라 임시적이고 가변적인 값이 될 수 있으므로
                #차의 속성이 아닌 지역 변수와 리스트에 보관해주도록 한다.
                dist = car.route[-2].distanceTo(item)
                dist_time = dist / car_speed   #km/h 기준
                item_loading_time = (loading_unloading_time /60) * len(item.customers_list)
                total_time = dist_time + item_loading_time
                car_times = [car, item, dist, total_time]
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

        def base_cost(new_cars):
            new_cars.cost_list = []
            for i in unloaded:
                for j in range(len(total_cost)):
                    if total_cost[j][0] == i:
                        new_cars.cost_list.append([i, new_cars.dist_cost + total_cost[j][1]])
            sorted_dist = []
            for i in range(len(new_cars.cost_list)):
                if len(new_cars.route) >= 3:
                    sorted_dist.append((i, haversine(new_cars.route[-2].position, unloaded[i].position)))
                    new_cars.cost_list[i][1] += haversine(new_cars.route[-2].position, unloaded[i].position)
            sorted_dist = sorted(sorted_dist, key=lambda x : x[1])
            for i in range(len(sorted_dist)):
                for j in range(len(new_cars.cost_list)):
                    if sorted_dist[i][0] == new_cars.cost_list[j][0]:
                        new_cars.cost_list[j][1] += sorted_dist[i][1] * i * cost_car_customer_var        
            new_cars.cost_list = sorted(new_cars.cost_list, key=lambda x : x[1], reverse=False)

        

        #사전에 cost를 다 정리하도록 함.
        dist_customer_hub = [haversine(hub.position ,i.position) for i in customers]
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
        #print("regular_car",regular_car)
        for i in range(len(regular_car)):
            predict_car_num.append(list(itertools.combinations((regular_car), i+1)))

        for i in range(len(predict_car_num)):
            for j in range(len(predict_car_num[i])):
                total_car_cap = 0
                for k in range(len(predict_car_num[i][j])):
                    total_car_cap += predict_car_num[i][j][k].cap
                if total_car_cap >= tq:
                    new_predict.append(predict_car_num[i][j])



        #구역 검사
        all_latitude_list = []
        all_longitude_list = []
        for c in range(len(customers)):
            all_latitude_list.append(customers[c].position[0])
            all_longitude_list.append(customers[c].position[1])

        customer_points = MultiPoint([*zip(all_latitude_list, all_longitude_list)])
        points_convex_hull = customer_points.convex_hull
        convex_hull_bounds = points_convex_hull.bounds
        min_x = convex_hull_bounds[0]
        min_y = convex_hull_bounds[1]
        max_x = convex_hull_bounds[2]     
        max_y = convex_hull_bounds[3]
        print(customer_points)  #-> EMPTY
        print(all_latitude_list)    # -> [] 
        #print(min_x, min_y, max_x, max_y) #-> 다 nan값

        border_lines = points_convex_hull.boundary
        x_line2 = LineString([(min_x, all_longitude_list[all_latitude_list.index(min_x)]), hub.position,(max_x, all_longitude_list[all_latitude_list.index(max_x)])])
        y_line2 = LineString([(all_latitude_list[all_longitude_list.index(min_y)], min_y), hub.position,(all_latitude_list[all_longitude_list.index(max_y)], max_y)])
        xy_line = line_merge(x_line2, y_line2)
        yx_line = line_merge(y_line2, x_line2)
        unioned = border_lines.union(xy_line)
        unioned = unioned.union(yx_line)

        border_xy_lines = polygonize([unioned])
        polys = MultiPolygon(border_xy_lines)
        #폴리곤 지정 차량
        count_points_in_poly = []
        for i in range(len(polys.geoms)):
            count_points_in_poly.append((polys.geoms[i], [], i))
        
        #각 폴리곤에 몇 개의 점이 포함되어있는지 검사
        for cp in range(len(customer_points.geoms)):
            for pl in range(len(polys.geoms)):
                if customer_points.geoms[cp].covered_by(polys.geoms[pl]) == True:
                    for cpip in range(len(count_points_in_poly)):
                        if count_points_in_poly[cpip][0] == polys.geoms[pl]:
                            count_points_in_poly[cpip][1].append(customer_points.geoms[cp])

        count_points_in_poly = sorted(count_points_in_poly, key = lambda x : len(x[1]))                
        #생각해보니 new_cars의 개수와 폴리곤의 개수가 맞지 않을 수가 있다. -> 나중에 체크해보도록하자.

        #여러 조합이 들어올텐데 그중에서도 가장 짧은 길이를 가진 것을 고름.
        #어차피 1개의 조합부터 차례대로 들어온 것이므로 [0]을 고르면 된다.
        #-> 원래 처음에는 가장 차를 적게 쓰기 위해서 가장 작은 길이를 가진 [0]을 고른거였는데
        #수정하다보니까 알고리즘이 살짝 바뀌면서 가장 적은 길이를 가진 것을 고르는 것도 물론 중요하지만? 미적인... 그런것을 조금 더 보기 위해서~
        #폴리곤의 개수에 맞춰서 차를 맞출겁니다. 그니까? 폴리곤이 4개면? 차 개수도 4개인거임.
        
        #수량 코스트와 허브 반경 코스트는 루프 내에서 변하지 않는 것이다.
        #근데 아직 최소로 필요한 차의 개수에 맞춰 물품을 효율적으로 넣는 단계는 아니므로
        #나중에 적용시키도록 함.
        #print(new_predict)
        pre_car = []
        new_cars = list(new_predict[0])
        for np in range(len(new_predict)):
            if len(new_predict[np]) == len(polys.geoms):
                new_cars = list(new_predict[np])
                break
            
    
        for i in range(len(regular_car)):
            if regular_car[i] not in new_cars:
                pre_car.append(regular_car[i])

        new_cars = sorted(new_cars, key = lambda x: x.cap)
        for nc in range(len(new_cars)):
            for pll in range(len(count_points_in_poly)):
                if nc == pll:
                    #인덱스 번호
                    new_cars[nc].area = count_points_in_poly[pll][2]

        #점이 어떤 폴리곤에 포함되는지를 검사
        def check_area(i):
            for p in range(len(customer_points.geoms)):
                if customer_points.geoms[p].coords[0] == i.position:
                    for pl in range(len(polys.geoms)):
                        if customer_points.geoms[p].covered_by(polys.geoms[pl]) == True:
                            return pl   #인덱스 번호
        

        #이제 루프를 돌면서 물품을 넣을 차례

        unloaded_total_cbm = 0
        for u in unloaded:
            unloaded_total_cbm += u.cbm

        while len(unloaded) > 0:
            #차량마다 cost_list를 가지고 갱신된다.
            #차량의 수 만큼 갱신을 해주기 위해서 반복
            for c in range(len(new_cars)):
                new_cars[c].cost_list = []
                for i in unloaded:
                    for j in range(len(total_cost)):
                        if total_cost[j][0] == i:
                            area_number = check_area(i)
                            if area_number == new_cars[c].area: #-> 물품이 속하는 폴리곤 넘버와 차량이 지정된 폴리곤 넘버가 같으면? cost X
                               print(i, "이 속하는 폴리곤 넘버: ", area_number, "차량의 폴리곤 넘버: ", new_cars[c], new_cars[c].area,"구역")
                               new_cars[c].cost_list.append([i, new_cars[c].dist_cost + total_cost[j][1]]) 
                            else:
                                #넘버가 서로 같지 않으면 cost 부여.
                                new_cars[c].cost_list.append([i, new_cars[c].dist_cost + total_cost[j][1] + 10000])
                                print(i,"이 속하는 폴리곤 넘버: ", area_number, "차량의 폴리곤 넘버: ", new_cars[c], new_cars[c].area,"구역")
                sorted_dist = []
                for i in range(len(new_cars[c].cost_list)):
                    if len(new_cars[c].route) >= 3:
                        sorted_dist.append((i, haversine(new_cars[c].route[-2].position, unloaded[i].position)))
                        new_cars[c].cost_list[i][1] += haversine(new_cars[c].route[-2].position, unloaded[i].position)
                sorted_dist = sorted(sorted_dist, key=lambda x : x[1])
                for i in range(len(sorted_dist)):
                    for j in range(len(new_cars[c].cost_list)):
                        if sorted_dist[i][0] == new_cars[c].cost_list[j][0]:
                            new_cars.cost_list[j][1] += sorted_dist[i][1] * i * cost_car_customer_var        
                new_cars[c].cost_list = sorted(new_cars[c].cost_list, key=lambda x : x[1], reverse=False)
            #갱신하고 정렬까지 해주었음.
            #이제 값을 서로 비교해가면서 넣어주면 된다.
            
            #각 차량의 cost_list 중 가장 작은 물품이 겹칠 경우에는 cost값을 비교한다.
            #그중 최소가 되는 차량에 물품을 넣음.

            item_list = []
            car_item_list = []
            for k in range(len(new_cars)):
                item_list.append(new_cars[k].cost_list[0][0])
                car_item_list.append((new_cars[k], new_cars[k].cost_list[0]))
            print("각 차에서 코스트가 최소인 물품: ",car_item_list)
            
        
            #일단 절대적인 unloaded 리스트가 존재한다.
            #차마다 물품 리스트를 만들어서 해봤으나 1이라는 물품이 A차에서는 코스트가 높은데
            #B차에서는 코스트가 낮아서 알고리즘 상 B차에 넣는 게 맞다.
            #하지만 A차가 간 거리가 많거나 이미 물건이 많아서 1물품과의 거리가
            #B차보다 가까워도 기본적인 차의 코스트가 많아 코스트가 많이 부여가 되어
            #우선순위에서 밀리는 경우도 있다.
            #그런 경우를 최소로 하고 관리도 쉽고 직관적으로 만들기 위해서
            #고유의 unloaded 리스트가 존재하는 동시에 1이라는 물품에 대해
            #각 차들에 싣는 경우 그 cost를 계산하여 최소의 cost를 만들어 내는 차에 싣는다.

            #여기서부터는 기본적인 코스트가 추가된 물품을 서로 비교하며 넣어주도록 한다.
            #처음에는 중복인 물품과 아닌 물품을 따로 구분하고
            #차마다 물품 리스트를 만들어 서로 비교하여 넣어주었으나
            #만들다 보니 구조도 살짝 다르게 되었으며 관리하기도 어렵고 직관적이지 않아
            #따로 구분하지 않고 넣어보려고 한다.

            item_list = list(set(item_list))
            print("중복을 제거한 넣어줄 물품: ", item_list)

            next_item_time = []
            available_car_item_list = []
            car_times = []
            
            #현재 가능한 new_cars 중에서 다음 물품을 실을 수 있는지 검사함.
            #해당 코드들의 목적은 새로운 차를 배정을 해주기 위함임.
            for i in range(len(item_list)):
                for c in new_cars:
                    next_item_time.append(time_condition(c, item_list[i], car_times))

            for nit in range(len(next_item_time)):
                print(next_item_time[nit][0],"차량에 해당 아이템",next_item_time[nit][1], "을 넣을 수 있는지 검사.")
                if no_over_cap == True:
                    #과적 X
                    if next_item_time[nit][0].real_cbm + next_item_time[nit][1].cbm <= (next_item_time[nit][0].cap * max_capacity_cost):
                        if no_over_time == True:
                            #과적 X, 초과 근무 X
                            if next_item_time[nit][0].travel_time + next_item_time[nit][3] <= next_item_time[nit][0].whole_travel_time:
                                available_car_item_list.append(next_item_time[nit][1])
                                print("가능")
                            else:   print("차량의 근무 시간이 전체 시간보다 많아서 불가능", "차량 시간:",next_item_time[nit][0].travel_time, next_item_time[nit][1],"을 가는데 걸리는 시간: ", next_item_time[nit][3])
                        else:
                            #초과 근무가 가능하다고 하더라도 극단적인 현상을 피하기 위해 어느정도의 선을 그음.
                            if next_item_time[nit][0].travel_time + next_item_time[nit][3] <= next_item_time[nit][0].whole_travel_time + limit_time:
                                available_car_item_list.append(next_item_time[nit][1])
                                print("가능")
                            else:   print("초과근무 O 여도 시간을 넘어서 불가능","차량 시간:",next_item_time[nit][0].travel_time, next_item_time[nit][1],"을 가는데 걸리는 시간: ", next_item_time[nit][3])
                    else:   print("용량을 넘어서 불가능", next_item_time[nit][0],"차량 현재 용량:",next_item_time[nit][0].real_cbm, next_item_time[nit][1],"의 용량:", next_item_time[nit][1].cbm, "차량의 정해진 용량: ", next_item_time[nit][0].cap)

                else:
                    #cap 용량까지 과적이 허용된다고 하면
                    if next_item_time[nit][0].real_cbm + next_item_time[nit][1].cbm <= next_item_time[nit][0].cap - limit_capacity:
                        if no_over_time == True:
                            #과적 허용 O, 초과 근무 X
                            if next_item_time[nit][0].travel_time + next_item_time[nit][3] <= next_item_time[nit][0].whole_travel_time:
                                available_car_item_list.append(next_item_time[nit][1])
                                print("가능")
                            else:   print("과적이 가능하다고 해도 시간이 넘어서 불가능","차량 시간:",next_item_time[nit][0].travel_time, next_item_time[nit][1],"을 가는데 걸리는 시간: ", next_item_time[nit][3])
                        else:
                            if next_item_time[nit][0].travel_time + next_item_time[nit][3] <= next_item_time[nit][0].whole_travel_time + limit_time:
                                available_car_item_list.append(next_item_time[nit][1])
                                print("가능")
                            else:   print("과적과 초과근무가 허용되도 시간이 넘어서 불가능","차량 시간:",next_item_time[nit][0].travel_time, next_item_time[nit][1],"을 가는데 걸리는 시간: ", next_item_time[nit][3])
                    else:   print("용량을 넘어서 불가능",next_item_time[nit][0],"차량 현재 용량:",next_item_time[nit][0].real_cbm, next_item_time[nit][1],"의 용량:", next_item_time[nit][1].cbm, "차량의 정해진 용량: ", next_item_time[nit][0].cap)

            
            

            if len(available_car_item_list) == 0:
                print("가능한 차량이 없으므로 예비 차량 중에서 조건에 맞는 차량 선택")
                no_available_item = item_list
                available_car_time_condition = []
                available_car = []
                no_available_item = list(set(no_available_item))
                print("적재 불가능한 아이템: ", no_available_item)

                car_times = []
                for nai in no_available_item:
                    for pc in pre_car:
                        #이 차마다 time_condition !
                        available_car_time_condition.append(time_condition(pc, nai, car_times))

                for actc in available_car_time_condition:
                    if no_over_cap == True:
                        if (actc[0].cap * max_capacity_cost) >= actc[1].cbm:
                            if no_over_time == True:
                                if actc[0].whole_travel_time >= actc[0].travel_time + actc[3]:
                                        available_car.append([actc[0], actc[1]])
                            else:
                                if actc[0].travel_time + actc[3] <= actc[0].whole_travel_time + limit_capacity:
                                    available_car.append([actc[0], actc[1]])
                    else:
                        #과적 금지
                        if actc[0].real_cbm + actc[1].cbm <= actc[0].cap:
                            if no_over_time == True:
                                #과적금지, 초과근무 X
                                if actc[0].travel_time + actc[3] <= actc[0].whole_travel_time:
                                    available_car.append([actc[0], actc[1]])
                            else:
                                #과적금지, 초과근무 O
                                if actc[0].travel_time + actc[3] <= actc[0].whole_travel_time + limit_capacity:
                                    available_car.append([actc[0], actc[1]])
                        
                print("예비 차들 중에서 물품 중 적재가 불가능 했던", no_available_item,"을 넣을 수 있는 차의 리스트: ")
                print(available_car)


                if len(available_car) != 0:
                    for nai in range(len(no_available_item)):
                        for ac in range(len(available_car)):
                            if available_car[ac][1] == no_available_item[nai]:
                                if available_car[ac][0] not in new_cars:
                                    new_cars.append(available_car[ac][0])
                                    base_cost(available_car[ac][0])
                                    print("추가한 차: ", available_car[ac][0], "한계 용량: ", available_car[ac][0].cap, "제한 용량: ", available_car[ac][0].cap * max_capacity_cost)
                                    pre_car.remove(available_car[ac][0])
                                    break
                        

            else:
                for cil in range(len(available_car_item_list)):
                    if available_car_item_list[cil] not in item_list :

                        no_available_item = []
                        available_car_time_condition = []
                        available_car = []
                        #item_list안의 모든 item이 적재가 되지 않거나 그 중 몇 개만 되거나 하는 경우이다.
                        print(item_list, available_car_item_list)
                        print("아이템 적재 불가능")
                        #일단 적재가 불가능한 아이템을 찾는다.
                        for acil in available_car_item_list:
                            for il in item_list:
                                if acil != il:
                                    no_available_item.append(il)
                        #다음 물품에 넣을 차가 없을 경우 새로운 차를 추가함.
                        #물품을 적재할 수 있는 용량과 시간 조건이 맞는 차량을 골라서 추가한다. 그리고 추가할 차량이 여러 대 일 수 있다.
                        no_available_item = list(set(no_available_item))
                        print("적재 불가능한 아이템: ", no_available_item)
                        car_times = []
                        for nai in no_available_item:
                            for pc in pre_car:
                                #이 차마다 time_condition !
                                available_car_time_condition.append(time_condition(pc, nai, car_times))

                        for actc in available_car_time_condition:
                            if no_over_cap == True:
                                if (actc[0].cap * max_capacity_cost) >= actc[1].cbm:
                                    if no_over_time == True:
                                        if actc[0].whole_travel_time >= actc[0].travel_time + actc[3]:
                                            available_car.append([actc[0], actc[1]])
                                    else:
                                        if actc[0].travel_time + actc[3] <= actc[0].whole_travel_time + limit_capacity:
                                            available_car.append([actc[0], actc[1]])
                            else:
                                #과적 금지
                                if actc[0].real_cbm + actc[1].cbm <= actc[0].cap:
                                    if no_over_time == True:
                                        #과적금지, 초과근무 X
                                        if actc[0].travel_time + actc[3] <= actc[0].whole_travel_time:
                                            available_car.append([actc[0], actc[1]])
                                    else:
                                        #과적금지, 초과근무 O
                                        if actc[0].travel_time + actc[3] <= actc[0].whole_travel_time + limit_capacity:
                                            available_car.append([actc[0], actc[1]])
                        
                        print("예비 차들 중에서 물품 중 적재가 불가능 했던", no_available_item,"을 넣을 수 있는 차의 리스트: ")
                        print(available_car)


                        if len(available_car) != 0:
                            for nai in range(len(no_available_item)):
                                for ac in range(len(available_car)):
                                    if available_car[ac][1] == no_available_item[nai]:
                                        if available_car[ac][0] not in new_cars:
                                            new_cars.append(available_car[ac][0])
                                            base_cost(available_car[ac][0])
                                            print("추가한 차: ", available_car[ac][0], "한계 용량: ", available_car[ac][0].cap, "제한 용량: ", available_car[ac][0].cap * max_capacity_cost)
                                            pre_car.remove(available_car[ac][0])
                                            break
                
                    else:
                        print("적재 가능")
                        print(available_car_item_list)

            print("차 목록: ", new_cars)

            temp_solution = []
            delete_car_list = []
            if len(item_list) != 0:
                for i in range(len(item_list)):
                    for j in range(len(new_cars)):
                        #해당 물품에 대해서 각 차마다의 코스트를 정리
                        #하기 위해서 여러 조건들을 체크함.
                        
                        #여러 조건을 체크하기 전에 모든 물품들은 각 차의 한계 중량을 넘지 않아야 한다.
                        if new_cars[j].real_cbm + item_list[i].cbm <= new_cars[j].cap:

                            #최대 중량 체크
                            if new_cars[j].real_cbm + item_list[i].cbm > new_cars[j].cap * max_capacity_cost:
                                if no_over_cap == True:
                                    #과적 금지
                                    for cl in new_cars[j].cost_list:
                                        if cl[0] == item_list[i]:
                                            cl[1] = math.inf
                                else:
                                    over_capacity = ((new_cars[j].real_cbm + item_list[i].cbm) - (new_cars[j].cap * max_capacity_cost))
                                    print(new_cars[j],"가 최대 중량인", new_cars[j].cap*max_capacity_cost, "를 초과하여 COST를 추가 부여한다.")
                                    #변화된 COST와 차의 정보는 CAR의 속성인 cost_list에 기록한다.
                                    #cost_list의 구조 => [[아이템1, 코스트], [아이템2, 코스트] . . .]
                                    
                                    #cost_list에서 해당 아이템을 찾고 코스트를 추가한다.
                                    for cl in (new_cars[j].cost_list):
                                        if cl[0] == item_list[i]:
                                            cl[1] += max_capacity_cost_var * over_capacity

                            #교차 검사
                            new_cars[j].item_cross_count = 0
                            for rc in range(len(new_cars)):
                                if len(new_cars[rc].route) >= 3:
                                    for rc_route in range(len(new_cars[rc].route) - 2):
                                        if intersect_line(new_cars[rc].route[rc_route].position, new_cars[rc].route[rc_route + 1].position, new_cars[j].route[-2].position, item_list[i].position) == True:
                                            #교차라면
                                            #다른 차의 경로들 중 얼마나 겹치는 지 count
                                            new_cars[j].item_cross_count += 1
                                            print(new_cars[j], "가", new_cars[rc], "의", new_cars[rc].route[rc_route], ",", new_cars[rc].route[rc_route + 1], "의 선분과 겹치며")
                                            print(new_cars[j], "가", item_list[i], "에 간다면 교차 횟수는", new_cars[j].item_cross_count, "이며, 결정된 교차는 ", new_cars[j].cross_count, "이다.")
                                            
                                    for cl in (new_cars[j].cost_list):
                                        if cl[0] == item_list[i]:
                                            cl[1] += ((new_cars[j].cross_count * al_cross_var) + (new_cars[j].item_cross_count * crossline_var))
                                            #cl[1] += (new_cars[j].item_cross_count * crossline_var)

                            #폴리곤 구역 검사
                            area_number = check_area(item_list[i])
                            if area_number == new_cars[j].area: #-> 물품이 속하는 폴리곤 넘버와 차량이 지정된 폴리곤 넘버가 같으면? cost X
                                print(item_list[i], "이 속하는 폴리곤 넘버: ", area_number, "차량의 폴리곤 넘버: ", new_cars[j], new_cars[j].area,"구역")
                                print("구역이 같으므로 코스트 부여 X")
                            else:
                                #넘버가 서로 같지 않으면 cost 부여.
                                for cl in (new_cars[j].cost_list):
                                    if cl[0] == item_list[i]:
                                    #cl[1] += ((new_cars[j].cross_count * al_cross_var) + (new_cars[j].item_cross_count * crossline_var))
                                        cl[1] += 30000
                                        print(item_list[i],"이 속하는 폴리곤 넘버: ", area_number, "차량의 폴리곤 넘버: ", new_cars[j], new_cars[j].area,"구역")
                                        print("구역이 같지 않으므로 코스트 30000 부여")

                            
                            #시간 초과 검사
                            car_times = []
                            car_time = time_condition(new_cars[j], item_list[i], car_times)
                            if car_time != None:
                                print("시간 초과 검사중: ", new_cars[j], item_list[i], car_time)
                                if new_cars[j].travel_time + car_time[3] > new_cars[j].whole_travel_time:
                                    if no_over_time == True:
                                        #초과 근무 불가
                                        for cl in new_cars[j].cost_list:
                                            if cl[0] == item_list[i]:
                                                cl[1] = math.inf
                                    else:
                                        late_time = ((new_cars[j].travel_time + car_time[3]) - new_cars[j].whole_travel_time) * 60
                                        print(new_cars[j], "차량이 근무시간", new_cars[j].whole_travel_time, "에서", late_time, "만큼 초과한다.")

                                        for cl in (new_cars[j].cost_list):
                                            if cl[0] == item_list[i]:
                                                cl[1] += late_time * timewindow_cost_var
                                else:
                                    print(new_cars[j], "가", new_cars[j].travel_time, "시간을 사용했으며", item_list[i], "를 넣어도 시간을 초과하지 않는다.")
                            
                            #조건이 맞는 범위 안에서 temp_solution에 집어넣는다.
                            #temp_solution은 임시적인 해답 리스트로 코스트가 최저인 차와 아이템의 조합을 넣는다.
                            #이 리스트의 구조는 [[차량1, 코스트]. [차량2, 코스트] . . .]
                            #아이템의 코스트 추가가 끝나고 나서 하나씩 넣는 것도 문제가 없어 보여 정렬하기 편하라고 선정됨.]
                            for w in range(len(new_cars[j].cost_list)):

                                if new_cars[j].cost_list[w][0] == item_list[i] and new_cars[j].cost_list[w][1] != math.inf:
                                    temp_solution.append([new_cars[j], new_cars[j].cost_list[w][1], item_list[i]])
                                
                                #math.inf는 어차피 초과 근무 금지일 경우에만 나타나는 값이다.
                                #초과 근무를 할 가능성이 있는 차량을 삭제 리스트에 넣는다.
                                if new_cars[j].cost_list[w][1] == math.inf:
                                    delete_car_list.append(new_cars[j])

                        else:
                            #한계 중량을 넘는다면
                            #pre_car에서 새로운 차를 배정한다.       
                            for pc in range(len(pre_car)):
                                if pre_car[pc].real_cbm + item_list[i].cbm <= pre_car[pc].cap:
                                    new_cars.append(pre_car[pc])
                                    break
                    
                    #삭제
                    #근데 냅다 삭제해버리면 안되고
                    #현재 차량의 남은 용량과 소요 시간을 보고 어느정도 삭제할 수 있다라는 정도면 삭제고
                    #그게 아니라면 삭제하지 않는다.
                    if len(delete_car_list) != 0:
                        print("remove_list: ", delete_car_list)
                        real_del_list = []
                        
                        for dc in range(len(delete_car_list)):
                            #삭제할지 말지 본다.
                            if no_over_cap == True:
                                #과적 금지면
                                if no_over_time == True:
                                    #과근 X
                                    if delete_car_list[dc].real_cbm >= (delete_car_list[dc].cap * max_capacity_cost) * used_limit_capacity:
                                        if delete_car_list[dc].travel_time >= delete_car_list[dc].whole_travel_time * used_limit_time:
                                            real_del_list.append(delete_car_list[dc])
                                else:
                                    if delete_car_list[dc].real_cbm >= (delete_car_list[dc].cap * max_capacity_cost) * used_limit_capacity:
                                        if delete_car_list[dc].travel_time >= delete_car_list[dc].whole_travel_time + limit_time:
                                            real_del_list.append(delete_car_list[dc])
                            else:
                                #과적 O
                                if no_over_time == True:
                                    #과근 X
                                    if delete_car_list[dc].real_cbm >= delete_car_list[dc].cap * used_limit_capacity:
                                        if delete_car_list[dc].travel_time >= delete_car_list[dc].whole_travel_time * used_limit_time:
                                            real_del_list.append(delete_car_list[dc])
                                else:
                                    #과근O
                                    if delete_car_list[dc].real_cbm >= delete_car_list[dc].cap * used_limit_capacity:
                                        if delete_car_list[dc].travel_time >= delete_car_list[dc].whole_travel_time + limit_time:
                                            real_del_list.append(delete_car_list[dc])

                            for rdl in range(len(real_del_list)):                             
                                if real_del_list[rdl] in new_cars:
                                    print("삭제할 차량: ",real_del_list[rdl])
                                    new_cars.remove(real_del_list[rdl])
                                
                    
                    #가능한 한 경우의 수를 줄여서 간소화된 루트를 만들어야 하므로
                    #물론 다 넣을 수 없으므로 차를 추가하는 것은 맞으나
                    #초기부터 차를 추가하여 경우의 수를 늘릴 경우 루트를 지저분하게 만들 가능성이 높음.
                    
                    #차량의 현 상태를 확인하고 아직 남아있는 물품을 사용할 수 있는 차량에 다 적재가 되는지 안되는지 확인
                    #또한 과적 금지, 초과 근무 금지 여부에 따라 비교하는 숫자가 다름.
                    #적재가 다 되지 않는다면 new_cars에 새로운 차량을 집어 넣어줌.
                    """
                    available_car_cbm = 0
                    available_car_travel_time = 0
                    for ac in new_cars:
                        if no_over_cap == True:
                            available_car_cbm += (ac.cap * max_capacity_cost) - ac.real_cbm
                        else:
                            available_car_cbm += (ac.cap - ac.real_cbm)
                        
                        #no_over_time == True가 아닐 경우에도 계산해주는 이유는 다음과 같다.
                        #아무리 초과 근무가 가능하다고 한들
                        #한 차의 travel_time이 극단적으로 많이 증가하는 경우를 피하기 위해서이다.
                        available_car_travel_time += (ac.whole_travel_time - ac.travel_time)
                    """
                    #1160파주 케이스에 특이 케이스 물품이 있음. 66
                    #이 TMS의 코드는 물품을 비슷한 장소에 있는 물품끼리 모아서 합치는 데 그게 바로 unloaded 이다.
                    #그래서 하나의 물품 66 이어도 굉장히 많은 시간이 소요될 수 있다.
                    #보통 근로자의 근무 시간은 엑셀 데이터 상 9am ~ 6pm 으로 점심시간을 포함한다고 하더라도 9시간 이다.
                    #66 하나만 가도 10시간이 넘어버리기 때문에 데이터를 수정하지 않는 한 사실 상 갈 수 있는 차가 없는 것이다.
                    #66의 customers_list를 이용하여 물품을 나누어서 넣을 수 있는 방법도 있겠으나 글로벌 시장에 초점을 둘 것이므로
                    #총 근무 시간을 늘리고 66의 물품을 줄이기로 결정하였다. 9시간 에서 14시간 으로 변경!

                    #미국과 같이 거리가 먼 장소를 가는 경우도 데이터를 수정하지 않는 한 no_over_time == True일 경우 갈 수 없음.
                    #미국은 운행시작과 종료시간을 0시와 12시로 정해 놓지 않고 시간 수를 쓴다
                    #이게 무슨 말이냐 -> 총 72시간 동안의 시간이 있다면 0시 부터 72시로 설정해놓았다.


                    """
                    #가능한 차량에 적재 가능한 용량보다 unloaded의 용량이 더 크다면 -> 새로운 차량을 배정
                    if ((available_car_cbm < unloaded_total_cbm) and len(new_cars) < 3):
                        print("적재 가능한 양: ", available_car_cbm, "unloaded의 총 양: ", unloaded_total_cbm, "차가 움직일 수 있는 시간: ", available_car_travel_time)
                        new_cars.append(pre_car[0])
                        pre_car.remove(pre_car[0])
                    """            

                    solution_list = sorted(temp_solution, key=lambda x: x[1], reverse=False)
                    print(item_list[i], "의 Solution_list: ", solution_list)
                    solution = solution_list[0]
                    print(item_list[i], "의 Solution: ", solution[0], "에", item_list[i], "을 넣어준다.", solution[0].travel_time)
                    assign(solution[0], item_list[i], solution[1])
                    print("assign 후: ",solution[0].real_cbm, solution[0].travel_time)                  
                    temp_solution = []

                #이 밑의 코드는 아이템 개수에 따라서 [아이템1, [[차량1, 코스트], [차량2, 코스트] . . .], [아이템2, [[차량1, 코스트], [차량2, 코스트] . . .]]]
                #이러한 구조로 만든 것인데 정렬에 불편함이 있어 선정되지 않음.
                """
                temp_solution = []
                for q in range(len(item_list)):
                    temp_solution.append([item_list[q], []])
                    for w in range(len(new_cars)):
                        for e in range(len(new_cars[w].cost_list)):
                            if new_cars[w].cost_list[e][0] == item_list[q]:
                                temp_solution[q][1].append([new_cars[w], new_cars[w].cost_list[e][1]])
                print("temp_solution :", temp_solution)
                #solution = sorted(temp_solution, key=lambda x: x[1][])
                """    
            
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