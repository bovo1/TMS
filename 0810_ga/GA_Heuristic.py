from Component import * 
import itertools, random, time, math
class GA:
    def __init__(self, problem, args):
        self.problem = problem
        self.max_capacity = args[3]
        self.limit_capacity = args[4]
        self.limit_time = args[5]
        self.quantity_cost = args[0]
        self.hub_radius_cost = args[1]
        self.crossline_var = args[2]
        self.car_customer_var = args[14]
        self.max_capacity_cost = args[6]
        self.timewindow_cost = args[7]
        self.car_speed = args[8]
        self.loading_unloading_time = args[9]
        self.no_over_capacity = args[12]
        self.no_over_time = args[13]
        self.yes_lent_car = args[23]
        self.used_limit_time = args[15]
        self.used_limit_cap = args[16]
        self.count_limit = args[17]
        self.lent_car_cost = args[18]
        self.num_of_generation = args[19]
        self.no_update_count = args[20]
        self.mutant_percent = args[21]
        self.change_percnet = args[22]
        self.lent_car_percent = args[24]
        self.hub_radius = args[25]
        self.good_gene_relay_count = args[26]
        self.over_dist = args[28]
        self.hub = problem.hub
        if float(args[25]) == 0:
            dist_customer_hub = [haversine(self.hub.position, i.position) for i in problem.customers]
            self.hub_radius = sum(dist_customer_hub) / len(dist_customer_hub)
        self.gene_select_percent = args[27]
        self.not_filled_capacity = 2000


    #def three_opt(self, customers : list[Customer], i, j) -> list[Customer]:
    def three_opt(self, customers : list, i, j) -> list:
        k = len(customers)
        if i > 0:
            return customers[:i] + customers[j:i-1:-1] + customers[-1:-k+j:- 1]
        else:
            return customers[j:i:-1] + [customers[i]] + customers[-1:-(k-j):-1]

    #def local_search(self, customers : list[Customer]) -> list[Customer]:
    def local_search(self, customers : list) -> list:
        #단일루트 최적화
        #1. 배달고객 수 1명 이하면 바로 종료.
        #2. loop : 최적해 변화 없으면 종료.
            #3. loop2 : 모든 경우의수를 전부 서치하면 종료
                #4. 2명 고객 고르는 모든 경우의 수를 구한다.
                #5. 3opt을 실행하고, 현 최적해와 비교하고 갱신.
        if len(customers) <= 1:
            return customers 
        best_customers = customers
        best_distance = self.problem.route_distance(customers)

        while True:
            new_custs = []
            new_distances = []
            for i, j in itertools.combinations(range(len(best_customers)), 2):
                new_custs.append(self.three_opt(best_customers, i, j))
                new_distances.append(self.problem.route_distance(new_custs[-1]))
            local_best_distance = min(new_distances)
            if best_distance > local_best_distance :
                best_distance = local_best_distance 
                best_customers = new_custs[new_distances.index(best_distance)]
            else:
                return best_customers 

    def time_condition(self, item_list, dist):
        #이 거리는 경우의 수에 따라 임시적이고 가변적인 값이 될 수 있으므로
        #차의 속성이 아닌 지역 변수와 리스트에 보관해주도록 한다.
        item_loading_time = 0
        dist_time = dist / self.car_speed   #km/h 기준
        for i in range(len(item_list)):
            item_loading_time += (self.loading_unloading_time / 60 )* len(item_list[i].customers_list)
        
        total_time = dist_time + item_loading_time

        return total_time

    def time_condition2(self, item, dist):
        item_loading_time = 0
        dist_time = dist / self.car_speed
        item_loading_time += (self.loading_unloading_time / 60 )* len(item.customers_list)
        total_time = dist_time + item_loading_time

        return total_time

    def ccw(self, a, b, c):
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
    def intersect_line(self, a,b,c,d):
        if self.ccw(a,b,c) * self.ccw(a,b,d) < 0:
            #교차
            if self.ccw(c,d,a) * self.ccw(c,d,b) < 0:
                return True
        else:
            return False

    def initial_chromosomes(self, problem, num):
        cars = [i for i in problem.cars]
        owned_cars = [i for i in cars if i.id[0] != 'Y']
        lent_cars = [i for i in cars if i not in owned_cars]
        
        if self.yes_lent_car == True:
            available_cars = cars
            car_prob = [(1 - self.lent_car_percent)/len(owned_cars) for i in owned_cars] + [self.lent_car_percent/len(lent_cars) for i in lent_cars]
        else:
            available_cars = owned_cars
            car_prob = [1.0/len(available_cars) for i in available_cars]
            if len(available_cars) == 0:
                car_prob = [1.0/len(lent_cars) for i in lent_cars]
        
        self.available_car_list = available_cars
        
        car_cap = [i.cap for i in available_cars]
        print('car_cap: ', car_cap)
        
        #print(car_prob)
        customers =[c for c in problem.abst_c if c.cbm < max(car_cap)]
        unloaded = [c for c in problem.abst_c if c not in customers]
        chromosomes = []


        #염색체의 다양성을 유지하되 좋은 염색체의 이용이 더 잘 되도록 초기 염색체를 구성함.

        while len(chromosomes) < num:
            chrom = []
            car_item_list = [[car] for car in self.available_car_list]
            valid_cars = copy.copy(available_cars)
            valid_cars_cap = car_cap[:]
            valid_cars_prob = car_prob[:]
            for cust in customers:
                while True:
                    if len(valid_cars) == 0:
                        print(f'customer {cust} cbm : {cust.cbm}', valid_cars_cap)
                        break 

                    car = np.random.choice(valid_cars, size = 1, p = valid_cars_prob)[0] 
                    car_idx = valid_cars.index(car)

                    if len(chromosomes) < self.count_limit * 1:
                        #초기 염색체 개수의 70% 는 조건이 적용된 염색체 사용
                        #나머지 30%는 조건 적용이 확실히 안된 염색체로 다양하게 구성.
                        if self.no_over_capacity == True:
                            #과적 금지
                            if valid_cars_cap[car_idx] - cust.cbm >= car.cap - (car.cap * self.max_capacity):
                                for cil in range(len(car_item_list)):
                                    if car_item_list[cil][0] == car:
                                        car_time = self.time_condition2(cust, self.problem.route_distance(car_item_list[cil][1:]))

                                if self.no_over_time == True:
                                    if car_time < car.whole_travel_time:
                                        valid_cars_cap[car_idx] -= cust.cbm
                                        for cil in range(len(car_item_list)):
                                            if car_item_list[cil][0] == car:
                                                car_item_list[cil].append(cust)
                                        chrom.append(car)
                                        break
                                    else:
                                        valid_cars.remove(car)
                                        valid_cars_cap.remove(valid_cars_cap[car_idx])   
                                        valid_owned_cars = [i for i in valid_cars if i.id[0] != 'Y']
                                        if self.yes_lent_car == True:
                                            valid_owned_cars = [i for i in valid_cars if i.id[0] != 'Y']
                                            valid_lent_cars = [i for i in valid_cars if i.id[0] == 'Y']
                                            if len(valid_owned_cars) > 0 and len(valid_lent_cars) > 0:
                                                valid_cars_prob = ([(1 - self.lent_car_percent)/len(valid_owned_cars) for i in valid_owned_cars] +
                                                                        [self.lent_car_percent/len(valid_lent_cars) for i in valid_lent_cars])
                                            elif len(valid_owned_cars) == 0:
                                                valid_cars_prob = [1/len(valid_lent_cars) for i in valid_lent_cars]
                                            elif len(valid_lent_cars) == 0:
                                                valid_cars_prob = [1/len(valid_owned_cars) for i in valid_owned_cars] 

                                        else:
                                            valid_owned_cars = [i for i in valid_cars if i.id[0] != 'Y']
                                            valid_lent_cars = [i for i in cars if i.id[0] == 'Y']
                                            if len(valid_owned_cars) > 0 and len(valid_lent_cars) > 0:
                                                valid_cars_prob = ([1/len(valid_owned_cars) for i in valid_owned_cars])
                                            elif len(valid_owned_cars) == 0:
                                                valid_cars_prob = [1/len(valid_lent_cars) for i in valid_lent_cars]
                                                                                
                                else:
                                    #초과 근무 X
                                    if car_time < car.whole_travel_time + self.limit_time:
                                        valid_cars_cap[car_idx] -= cust.cbm
                                        for cil in range(len(car_item_list)):
                                            if car_item_list[cil][0] == car:
                                                car_item_list[cil].append(cust)
                                        chrom.append(car)
                                        break
                                    else:
                                        valid_cars.remove(car)
                                        valid_cars_cap.remove(valid_cars_cap[car_idx])   
                                        if self.yes_lent_car == True:
                                            valid_owned_cars = [i for i in valid_cars if i.id[0] != 'Y']
                                            valid_lent_cars = [i for i in valid_cars if i.id[0] == 'Y']
                                            if len(valid_owned_cars) > 0 and len(valid_lent_cars) > 0:
                                                valid_cars_prob = ([(1 - self.lent_car_percent)/len(valid_owned_cars) for i in valid_owned_cars] +
                                                                        [self.lent_car_percent/len(valid_lent_cars) for i in valid_lent_cars])
                                            elif len(valid_owned_cars) == 0:
                                                valid_cars_prob = [1/len(valid_lent_cars) for i in valid_lent_cars]
                                            elif len(valid_lent_cars) == 0:
                                                valid_cars_prob = [1/len(valid_owned_cars) for i in valid_owned_cars] 

                                        else:
                                            valid_owned_cars = [i for i in valid_cars if i.id[0] != 'Y']
                                            valid_lent_cars = [i for i in cars if i.id[0] == 'Y']
                                            if len(valid_owned_cars) > 0 and len(valid_lent_cars) > 0:
                                                valid_cars_prob = ([1/len(valid_owned_cars) for i in valid_owned_cars])
                                            elif len(valid_owned_cars) == 0:
                                                valid_cars_prob = [1/len(valid_lent_cars) for i in valid_lent_cars]
                            else:
                                #무게 조건에서 탈락
                                valid_cars.remove(car)
                                valid_cars_cap.remove(valid_cars_cap[car_idx])   

                                if self.yes_lent_car == True:
                                    valid_owned_cars = [i for i in valid_cars if i.id[0] != 'Y']
                                    valid_lent_cars = [i for i in valid_cars if i.id[0] == 'Y']
                                    if len(valid_owned_cars) > 0 and len(valid_lent_cars) > 0:
                                        valid_cars_prob = ([(1 - self.lent_car_percent)/len(valid_owned_cars) for i in valid_owned_cars] +
                                                                [self.lent_car_percent/len(valid_lent_cars) for i in valid_lent_cars])
                                    elif len(valid_owned_cars) == 0:
                                        valid_cars_prob = [1/len(valid_lent_cars) for i in valid_lent_cars]
                                    elif len(valid_lent_cars) == 0:
                                        valid_cars_prob = [1/len(valid_owned_cars) for i in valid_owned_cars] 

                                else:
                                    valid_owned_cars = [i for i in valid_cars if i.id[0] != 'Y']
                                    valid_lent_cars = [i for i in cars if i.id[0] == 'Y']
                                    if len(valid_owned_cars) > 0 and len(valid_lent_cars) > 0:
                                        valid_cars_prob = ([1/len(valid_owned_cars) for i in valid_owned_cars])
                                    elif len(valid_owned_cars) == 0:
                                        valid_cars_prob = [1/len(valid_lent_cars) for i in valid_lent_cars]
                        else:
                            #과적 허용
                            if valid_cars_cap[car_idx] - cust.cbm >= car.cap - (car.cap - self.limit_capacity):
                                for cil in range(len(car_item_list)):
                                    if car_item_list[cil][0] == car:
                                        car_time = self.time_condition2(cust, self.problem.route_distance(car_item_list[cil][1:]))
                                if self.no_over_time == True:
                                    if car_time < car.whole_travel_time:
                                        valid_cars_cap[car_idx] -= cust.cbm
                                        for cil in range(len(car_item_list)):
                                            if car_item_list[cil][0] == car:
                                                car_item_list[cil].append(cust)
                                        chrom.append(car)
                                        break
                                    else:
                                        valid_cars.remove(car)
                                        valid_cars_cap.remove(valid_cars_cap[car_idx])   
                                        #print("1조건이 안되서 차를 삭제하는 부분")
                                        if self.yes_lent_car == True:
                                            valid_owned_cars = [i for i in valid_cars if i.id[0] != 'Y']
                                            valid_lent_cars = [i for i in valid_cars if i.id[0] == 'Y']
                                            if len(valid_owned_cars) > 0 and len(valid_lent_cars) > 0:
                                                valid_cars_prob = ([(1 - self.lent_car_percent)/len(valid_owned_cars) for i in valid_owned_cars] +
                                                                        [self.lent_car_percent/len(valid_lent_cars) for i in valid_lent_cars])
                                            elif len(valid_owned_cars) == 0:
                                                valid_cars_prob = [1/len(valid_lent_cars) for i in valid_lent_cars]
                                            elif len(valid_lent_cars) == 0:
                                                valid_cars_prob = [1/len(valid_owned_cars) for i in valid_owned_cars] 

                                        else:
                                            valid_owned_cars = [i for i in valid_cars if i.id[0] != 'Y']
                                            valid_lent_cars = [i for i in cars if i.id[0] == 'Y']
                                            if len(valid_owned_cars) > 0 and len(valid_lent_cars) > 0:
                                                valid_cars_prob = ([1/len(valid_owned_cars) for i in valid_owned_cars])
                                            elif len(valid_owned_cars) == 0:
                                                valid_cars_prob = [1/len(valid_lent_cars) for i in valid_lent_cars]
                                        #print(len(valid_cars), len(valid_cars_cap), len(valid_cars_prob))
                                                                                
                                else:
                                    #초과 근무 X
                                    if car_time < car.whole_travel_time + self.limit_time:
                                        #print('car_time: ', car_time)
                                        valid_cars_cap[car_idx] -= cust.cbm
                                        for cil in range(len(car_item_list)):
                                            if car_item_list[cil][0] == car:
                                                car_item_list[cil].append(cust)
                                        chrom.append(car)
                                        break
                                    else:
                                        #print("2조건이 안되서 차를 삭제하는 부분")
                                        valid_cars.remove(car)
                                        valid_cars_cap.remove(valid_cars_cap[car_idx]) 
                                        

                                        if self.yes_lent_car == True:
                                            valid_owned_cars = [i for i in valid_cars if i.id[0] != 'Y']
                                            valid_lent_cars = [i for i in valid_cars if i.id[0] == 'Y']
                                            if len(valid_owned_cars) > 0 and len(valid_lent_cars) > 0:
                                                valid_cars_prob = ([(1 - self.lent_car_percent)/len(valid_owned_cars) for i in valid_owned_cars] +
                                                                        [self.lent_car_percent/len(valid_lent_cars) for i in valid_lent_cars])
                                            elif len(valid_owned_cars) == 0:
                                                valid_cars_prob = [1/len(valid_lent_cars) for i in valid_lent_cars]
                                            elif len(valid_lent_cars) == 0:
                                                valid_cars_prob = [1/len(valid_owned_cars) for i in valid_owned_cars] 

                                        else:
                                            valid_owned_cars = [i for i in valid_cars if i.id[0] != 'Y']
                                            valid_lent_cars = [i for i in cars if i.id[0] == 'Y']
                                            if len(valid_owned_cars) > 0 and len(valid_lent_cars) > 0:
                                                valid_cars_prob = ([1/len(valid_owned_cars) for i in valid_owned_cars])
                                            elif len(valid_owned_cars) == 0:
                                                valid_cars_prob = [1/len(valid_lent_cars) for i in valid_lent_cars]
                                        #print(len(valid_cars), len(valid_cars_cap), len(valid_cars_prob))  
                                            
                            else:
                                #무게 조건에서 탈락
                                valid_cars.remove(car)
                                valid_cars_cap.remove(valid_cars_cap[car_idx])   
                                #print("3조건이 안되서 차를 삭제하는 부분")
                                if self.yes_lent_car == True:
                                    valid_owned_cars = [i for i in valid_cars if i.id[0] != 'Y']
                                    valid_lent_cars = [i for i in valid_cars if i.id[0] == 'Y']
                                    if len(valid_owned_cars) > 0 and len(valid_lent_cars) > 0:
                                        valid_cars_prob = ([(1 - self.lent_car_percent)/len(valid_owned_cars) for i in valid_owned_cars] +
                                                                [self.lent_car_percent/len(valid_lent_cars) for i in valid_lent_cars])
                                    elif len(valid_owned_cars) == 0:
                                        valid_cars_prob = [1/len(valid_lent_cars) for i in valid_lent_cars]
                                    elif len(valid_lent_cars) == 0:
                                        valid_cars_prob = [1/len(valid_owned_cars) for i in valid_owned_cars] 

                                else:
                                    valid_owned_cars = [i for i in valid_cars if i.id[0] != 'Y']
                                    valid_lent_cars = [i for i in cars if i.id[0] == 'Y']
                                    if len(valid_owned_cars) > 0 and len(valid_lent_cars) > 0:
                                        valid_cars_prob = ([1/len(valid_owned_cars) for i in valid_owned_cars])
                                    elif len(valid_owned_cars) == 0:
                                        valid_cars_prob = [1/len(valid_lent_cars) for i in valid_lent_cars]
                                #print(len(valid_cars), len(valid_cars_cap), len(valid_cars_prob), valid_cars_prob) 

                    else:
                        if valid_cars_cap[car_idx] - cust.cbm > 0 :
                            valid_cars_cap[car_idx] -= cust.cbm 
                            chrom.append(car)
                            break 
                        else:
                            if valid_cars_cap[car_idx] <= 1:
                                valid_cars.remove(car)
                                valid_cars_cap.remove(valid_cars_cap[car_idx])
                            #print("4조건이 안되서 차를 삭제하는 부분")
                            if self.yes_lent_car == True:
                                valid_owned_cars = [i for i in valid_cars if i.id[0] != 'Y']
                                valid_lent_cars = [i for i in valid_cars if i.id[0] == 'Y']
                                if len(valid_owned_cars) > 0 and len(valid_lent_cars) > 0:
                                    valid_cars_prob = ([(1 - self.lent_car_percent)/len(valid_owned_cars) for i in valid_owned_cars] +
                                                                [self.lent_car_percent/len(valid_lent_cars) for i in valid_lent_cars])
                                elif len(valid_owned_cars) == 0:
                                    valid_cars_prob = [1/len(valid_lent_cars) for i in valid_lent_cars]
                                elif len(valid_lent_cars) == 0:
                                    valid_cars_prob = [1/len(valid_owned_cars) for i in valid_owned_cars] 
                                        
                            else:
                                valid_owned_cars = [i for i in valid_cars if i.id[0] != 'Y']
                                valid_lent_cars = [i for i in cars if i.id[0] == 'Y']
                                if len(valid_owned_cars) > 0 and len(valid_lent_cars) > 0:
                                    valid_cars_prob = ([1/len(valid_owned_cars) for i in valid_owned_cars])
                                elif len(valid_owned_cars) == 0:
                                    valid_cars_prob = [1/len(valid_lent_cars) for i in valid_lent_cars]   
                            #print(len(valid_cars), len(valid_cars_cap), len(valid_cars_prob))  
                    
            if len(chrom) == len(customers):
                chromosomes.append(chrom)

        return chromosomes, unloaded

    def measure_fitness(self, chromosomes, first = False):
        customers = self.problem.abst_c
        #거리 O, 시간 O, 물량O, 교차O, 허브반경O, 차와 지점간의 거리O,
        #최대 중량비용 등...
        distances = []
        times = []
        quantities = []
        total_cross_cost = []
        hub_radiuses = []
        lack_time_list = []
        not_filled_list = []
        car_item_dist = []
        #print("염색체: ",chromosomes)
        for chrom in chromosomes:
            #차 - 고객 구조로 이루어진 루트들 만들기
            dictionary = {car : [] for car in self.available_car_list}
            for i in range(len(chrom)):
                car = chrom[i]
                customer = customers[i]
                dictionary[car].append(customer)
            #print('DICTIONARY: ', dictionary)
            
            car_item_list = []
            for car in dictionary:
                #dictionary[car].insert(0, self.problem.hub)
                car_item_list.append([car, dictionary[car]])
            #print(car_item_list)
            #각 루트 거리 구해서 염색체 총 거리 구하기
            total_distance = 0
            car_time = 0    
            over_times = 0
            over_time = 0
            lack_times = 0
            cross_line_count = 0
            car_customer_dist = 0
            total_quantity = 0
            hub_radius_diff = 0
            hub_radius_diffs = 0
            not_filled_val = 0
            for i in range(len(car_item_list)):      
                car = car_item_list[i][0]
                cust = self.local_search(dictionary[car])
                if car.id[0] != 'Y':
                    total_distance += self.problem.route_distance(cust)
                else:
                    total_distance += self.lent_car_cost * self.problem.route_distance(cust)
                #print(total_distance)
                car_time = self.time_condition(cust, self.problem.route_distance(cust))
                for l in range(i+1, len(car_item_list)):
                    for j in range(len(car_item_list[i][1]) - 1):
                        #print('첫번째',car_item_list[i][1][j], car_item_list[i][1][j+1])
                        car_customer_dist += haversine(car_item_list[i][1][j].position, car_item_list[i][1][j+1].position)
                        for k in range(len(car_item_list[l][1]) - 1):
                            #print('두번째',car_item_list[l][1][k], car_item_list[l][1][k+1])
                            if len(car_item_list[i][1]) >= 2 and len(car_item_list[l][1]) >= 2:
                                if self.intersect_line(car_item_list[i][1][j].position, car_item_list[i][1][j+1].position, car_item_list[l][1][k].position, car_item_list[l][1][k+1].position) == True:
                                    #print("교차", car_item_list[i][0])
                                    cross_line_count += 1
                for z in range(len(car_item_list[i][1])):
                    if haversine(self.hub.position, car_item_list[i][1][z].position) > self.hub_radius:
                        hub_radius_diff = (haversine(self.hub.position, car_item_list[i][1][z].position) - self.hub_radius) * self.hub_radius_cost
                        #print(haversine(self.hub.position, car_item_list[i][1][z].position), self.hub_radius, (haversine(self.hub.position, car_item_list[i][1][z].position) - self.hub_radius), (haversine(self.hub.position, car_item_list[i][1][z].position) - self.hub_radius) * self.hub_radius_cost)
                        hub_radius_diffs += hub_radius_diff
                        #print("허브 반경 넘음", hub_radius_diff)
                        #print("총 허브반경 넘은 값:", hub_radius_diffs)

                #오히려 채우지 않으면 비용을 추가함 
                if self.no_over_capacity == True:
                    not_filled_cbm = (car_item_list[i][0].cap * self.max_capacity)
                    if len(car_item_list[i][1]) >= 1:
                        for a in range(len(car_item_list[i][1])):
                            not_filled_cbm -= car_item_list[i][1][a].cbm
                        if not_filled_cbm > (car_item_list[i][0].cap * self.max_capacity) * 0.3:    #self.used_limit_cap 이었는데 이 값이 낮아야 좋은 것임. 0.3으로 해보기.
                            total_quantity += not_filled_cbm * self.quantity_cost                 
                        elif not_filled_cbm < 0 :
                            #if chrom in chromosomes:
                                #chromosomes.remove(chrom)
                                #car_item_list.remove(car_item_list[i])
                            total_quantity += 10**6
                    else:
                        #print("좋아")
                        not_filled_val += not_filled_cbm * self.not_filled_capacity * 10000
                else:
                    not_filled_cbm = car_item_list[i][0].cap + self.limit_capacity
                    if len(car_item_list[i][1]) >= 1:
                        for a in range(len(car_item_list[i][1])):
                            not_filled_cbm -= car_item_list[i][1][a].cbm
                        if not_filled_cbm > (car_item_list[i][0].cap + self.limit_capacity) * 0.3:
                            total_quantity += not_filled_cbm * self.quantity_cost
                        elif not_filled_cbm < 0:
                            #if chrom in chromosomes:
                            #    chromosomes.remove(chrom)
                            #    car_item_list.remove(car_item_list[i])
                            total_quantity += 10**6
                    else:
                        not_filled_val += not_filled_cbm * self.not_filled_capacity * 10000

                #차의 현재 지점과 다음 지점간의 거리가 멀다면 코스트 추가.
                total_car_item_dist = 0
                for cid in range(len(car_item_list[i][1]) - 1):
                    ci_dist = haversine(car_item_list[i][1][cid].position, car_item_list[i][1][cid+1].position)
                    if ci_dist > self.hub_radius / 5:
                        #멀다.
                        total_car_item_dist += ci_dist
                #허브와의 거리(처음 허브에서 나갈 때만 고려했음.)
                if len(car_item_list[i][1]) >= 1:
                    ci_dist = haversine(self.hub.position, car_item_list[i][1][0].position)
                    if ci_dist > self.hub_radius / 5:
                        total_car_item_dist += ci_dist
                               

                if self.no_over_time == True:
                    #초과근무 금지
                    if car_time > car.whole_travel_time:
                        #over_time = (car_time - car.whole_travel_time) * self.timewindow_cost
                        #if chrom in chromosomes:
                            #chromosomes.remove(chrom)
                            #car_item_list.remove(car_item_list[i])
                        over_time = 10**5
                    else:
                        if car_time < car.whole_travel_time * self.used_limit_time:
                            lack_car_time = car_time - (car.whole_travel_time * self.used_limit_time)
                            lack_times += (lack_car_time * self.timewindow_cost) * -1

                else:
                    #제한 시간 근무 초과
                    if car_time > car.whole_travel_time + self.limit_time:
                        #over_time = (car_time - (car.whole_travel_time - self.limit_time)) * self.timewindow_cost
                        #if chrom in chromosomes:
                            #chromosomes.remove(chrom)
                            #car_item_list.remove(car_item_list[i])
                        over_time = 10**5
                        #print("제한 근무 초과: ",car, car_time, over_time, over_times)
                    else:
                        if car_time < (car.whole_travel_time + self.limit_time) * self.used_limit_time:
                            lack_car_time = car_time - ((car.whole_travel_time + self.limit_time) * self.used_limit_time)
                            lack_times += (lack_car_time * self.timewindow_cost) * -1

                over_times += over_time
                
            lack_time_list.append(lack_times)
            hub_radiuses.append(hub_radius_diff)
            times.append(over_times)
            distances.append(total_distance)
            quantities.append(total_quantity)
            total_cross_cost.append(cross_line_count*self.crossline_var*100)
            not_filled_list.append(not_filled_val)
            car_item_dist.append(total_car_item_dist * 10)


        Total_list = []
        for c in range(len(hub_radiuses)):
            total = hub_radiuses[c] + distances[c] + quantities[c] + total_cross_cost[c] + lack_time_list[c] + times[c]  + not_filled_list[c] + car_item_dist[c]
            #print(lack_time_list[c])
            Total_list.append(total)
            #Total_list.append(hub_radiuses[c] + times[c] + distances[c] + total_cross_cost[c])
        
        print("TOTAL LIST: ", Total_list)

        if first == True:
            self.fitness_constant = min(Total_list) * 3
            print("self.fitness_constant: ",self.fitness_constant) 
        #print("distance: ", distances)
        fitness = [(self.fitness_constant/i) for i in Total_list]
        
        return fitness

    def is_chrom_valid(self, chromosome):
        dictionary = {car : [] for car in self.available_car_list}
        cust = self.problem.abst_c
        for i in range(len(chromosome)):
            car = chromosome[i]
            c = cust[i]
            dictionary[car].append(c)

        for car in self.problem.cars:
            cap = car.cap 
            customers = dictionary[car] 
            total_cbm = 0 
            for customer in customers:
                total_cbm += customer.cbm 

            if total_cbm > cap:
                return False
        return True 

    def mutate(self, chromosome):
        chro = chromosome[:]
        car1 = np.random.choice(self.available_car_list, size = 1, replace = False)[0]
        
        i = np.random.choice(range(len(chromosome)), size = 1, replace = False)[0]
        chro[i] = car1
        return chro
                
    def generate_child_chromosomes(self, parent_chromosomes, parent_fitness, best_chromosome, best_fitness):
        one_valid = True 
        two_valid = True
        total_fitness = sum(parent_fitness)
        select_prob = [i/total_fitness for i in parent_fitness]
        child_chromosomes = []
        car_cap = [i.cap for i in self.available_car_list]
        customers =[c for c in self.problem.abst_c if c.cbm < max(car_cap)]
        
        #엘리트 주의 선택 + 룰렛 휠 선택
        for r in range(self.good_gene_relay_count):
            child_chromosomes.append(best_chromosome)

        while len(child_chromosomes) < len(parent_chromosomes):
            i, j = np.random.choice(range(len(parent_chromosomes)), size = 2, replace = False, p = select_prob)
            chrom1, chrom2 = parent_chromosomes[i], parent_chromosomes[j]

            #print('chrom1: ', chrom1)
            #print('chorm2: ', chrom2)
            child = []
            num_of_car = [[car, []] for car in self.available_car_list]
            chrom1_total_cbm = 0
            chrom2_total_cbm = 0
            chrom1_total_time = 0
            chrom2_total_time = 0
            #candidate_car = []
            #real_candidate_car = []
            child1 = []
            child2 = []
            for k in range(len(chrom1)):
                change = False
                candidate_car = []
                real_candidate_car = []
                if random.random() > self.change_percnet:    
                        #print('k: ',k)
                        if len(child) >= 1:                    
                            for i in range(len(child)):
                                if chrom1[k] == child[i]:
                                    chrom1_total_cbm += customers[i].cbm
                                    chrom1_total_time += self.time_condition2(customers[i], self.problem.route_distance([customers[i]]))
                                if chrom2[k] == child[i]:
                                    chrom2_total_cbm += customers[i].cbm
                                    chrom2_total_time += self.time_condition2(customers[i], self.problem.route_distance([customers[i]]))
                            candidate_car.append([chrom1[k], chrom1_total_cbm, chrom1_total_time])                        
                            candidate_car.append([chrom2[k], chrom2_total_cbm, chrom2_total_time])

                            for z in range(len(candidate_car)):
                                #print('candidate_car: ', candidate_car)
                                if self.no_over_capacity == True:
                                    if candidate_car[z][1] + customers[k].cbm < candidate_car[z][0].cap * self.max_capacity:
                                        #print("무게 조건 합격")
                                        for l in range(len(num_of_car)):
                                            if num_of_car[l][0] == chrom1[k]:
                                                if len(num_of_car[l][1]) != 0:
                                                    chrom1_time = self.time_condition2(customers[k], haversine(num_of_car[l][1][-1].position, customers[k].position))
                                                else:
                                                    chrom1_time = self.time_condition2(customers[k], haversine(self.hub.position, customers[k].position))

                                            if num_of_car[l][0] == chrom2[k]:
                                                if len(num_of_car[l][1]) != 0:
                                                    chrom2_time = self.time_condition2(customers[k], haversine(num_of_car[l][1][-1].position, customers[k].position))
                                                else:
                                                    chrom2_time = self.time_condition2(customers[k], haversine(self.hub.position, customers[k].position))                     
                                        if self.no_over_time == True:
                                            if candidate_car[z][0] == chrom1[k]:
                                                if candidate_car[z][2] + chrom1_time < chrom1[k].whole_travel_time:
                                                    #print("시간 조건 합격")
                                                    
                                                    #과적X, 초과근무X 일 때
                                                    real_candidate_car.append(candidate_car[z][0])                                                       
                                                    #change = True
                                            elif candidate_car[z][0] == chrom2[k]:
                                                if candidate_car[z][2] + chrom2_time < chrom2[k].whole_travel_time:
                                                    #print("시간 조건 합격")
                                                    #과적X, 초과근무X 일 때
                                                    real_candidate_car.append(candidate_car[z][0])                                                       
                                                    #change = True
                                                                                          
                                        else:
                                            if candidate_car[z][0] == chrom1[k]:
                                                if candidate_car[z][2] + chrom1_time < chrom1[k].whole_travel_time + self.limit_time:
                                                    real_candidate_car.append(candidate_car[z][0])                                                       
                                                    #change = True
                                            elif candidate_car[z][0] == chrom2[k]:
                                                if candidate_car[z][2] + chrom2_time < chrom2[k].whole_travel_time + self.limit_time:
                                                    real_candidate_car.append(candidate_car[z][0])                                                       
                                                    #change = True       
                                else:
                                      #과적 O                                   
                                    if candidate_car[z][1] + customers[k].cbm < candidate_car[z][0].cap - self.limit_capacity:
                                        for l in range(len(num_of_car)):
                                            if num_of_car[l][0] == chrom1[k]:
                                                if len(num_of_car[l][1]) != 0:
                                                    chrom1_time = self.time_condition2(customers[k], haversine(num_of_car[l][1][-1].position, customers[k].position))    
                                                else:
                                                    chrom1_time = self.time_condition2(customers[k], haversine(self.hub.position, customers[k].position))
                                            if num_of_car[l][0] == chrom2[k]:
                                                if len(num_of_car[l][1]) != 0:
                                                    chrom2_time = self.time_condition2(customers[k], haversine(num_of_car[l][1][-1].position, customers[k].position))    
                                                else:
                                                    chrom2_time = self.time_condition2(customers[k], haversine(self.hub.position, customers[k].position))              
                                        if self.no_over_time == True:
                                            if candidate_car[z][0] == chrom1[k]:
                                                if candidate_car[z][2] + chrom1_time < chrom1[k].whole_travel_time:
                                                    #과적X, 초과근무X 일 때
                                                    real_candidate_car.append(candidate_car[z][0])                                                       
                                                    #change = True
                                            elif candidate_car[z][0] == chrom2[k]:
                                                if candidate_car[z][2] + chrom2_time < chrom2[k].whole_travel_time:
                                                    #과적X, 초과근무X 일 때
                                                    real_candidate_car.append(candidate_car[z][0])                                                       
                                                    #change = True
                                                                                          
                                        else:
                                            if candidate_car[z][0] == chrom1[k]:
                                                if candidate_car[z][2] + chrom1_time < chrom1[k].whole_travel_time + self.limit_time:
                                                    real_candidate_car.append(candidate_car[z][0])                                                       
                                                    #change = True
                                            elif candidate_car[z][0] == chrom2[k]:
                                                if candidate_car[z][2] + chrom2_time < chrom2[k].whole_travel_time + self.limit_time:
                                                    real_candidate_car.append(candidate_car[z][0])                                                       
                                                    #change = True   

                        
                        else:
                            real_candidate_car.append(chrom1[k])
                            real_candidate_car.append(chrom2[k])
                            #print('처음 real_candidate_car: ', real_candidate_car)

                        if len(real_candidate_car) != 0:
                            #이중에서 골라 넣으면 된다.
                            #차를 가장 적게 쓰는 방향으로 한다.
                            #print('num_of_car: ', num_of_car)
                            if len(real_candidate_car) == 2:
                                for c in range(len(num_of_car)):
                                    if num_of_car[c][0] == real_candidate_car[0]:
                                        car1 = num_of_car[c][0]
                                        car_len = len(num_of_car[c][1])
                                        break
                                
                                for rcc in range(len(real_candidate_car)):
                                    for c in range(len(num_of_car)):
                                        if num_of_car[c][0] == real_candidate_car[rcc]:
                                            if car_len < len(num_of_car[c][1]):
                                                car_len = len(num_of_car[c][1])
                                                car1 = num_of_car[c][0]
                                #print('after car1: ', car1)
                                child.append(car1)
                                for c in range(len(num_of_car)):
                                    if num_of_car[c][0] == car1:
                                        num_of_car[c][1].append(customers[k])
                                change = True
                            elif len(real_candidate_car) == 1:
                                child.append(real_candidate_car[0])
                                for c in range(len(num_of_car)):
                                    if num_of_car[c][0] == real_candidate_car[0]:
                                        num_of_car[c][1].append(customers[k])
                                change = True
                            
                            #print('child: ', child)
                        else:
                            #조건에 만족하는 차가 없으므로
                            #적당한 것을 골라 채워준다. -> 차를 적게 쓰는 방향
                            #print('num_of_car: ', num_of_car)
 
                            for c in range(len(num_of_car)):
                                if num_of_car[c][0] == candidate_car[0][0]:
                                    car1 = num_of_car[c][0]
                                    car_len = len(num_of_car[c][1])
                                    break
                            #print(car1)
                            for rcc in range(len(candidate_car)):
                                for c in range(len(num_of_car)):
                                    if num_of_car[c][0] == candidate_car[rcc][0]:
                                        if car_len < len(num_of_car[c][1]):
                                            car_len = len(num_of_car[c][1])
                                            car1 = num_of_car[c][0] 
                            #print('car1: ', car1)     
                            for c in range(len(num_of_car)):
                                if num_of_car[c][0] == car1:
                                    num_of_car[c][1].append(customers[k])                       
                            child.append(car1)
                            change = True
                            #print('child: ', child)
                else:
                    if random.random() > 0.5:
                        child.append(chrom1[k])
                    else:
                        child.append(chrom2[k])

            #if random.random() > self.mutant_percent:
            #    child = self.mutate(child)

            if len(child_chromosomes) < len(parent_chromosomes):
                child_chromosomes.append(child)

            #print('child_chromosomes: ', child_chromosomes)


        return child_chromosomes

    def global_search(self, num_of_chrom, num_of_generations, args, problem : Problem):
        #1. 초기 부모 해집단 구성.
        #2. 초기 부모 해집단 fitness 측정(local search로 단일해 구하고 거리측정, fitness점수 부여)
        #3. 현 최적해 구성 
        #4.loop : 최적해 갱신x가 5번 연속 일어나면 or 100세대 이상
            #5. loop2 : 자식 해집단 크기가 부모 해집단 크기가 되면 종료 
                #6. 랜덤으로 2 부모 염색체 선택(선택률은 fitness 크기 비례)
                #7. 교차율 0.7로 자식 염색체 2명 구성
                #8. 0.001 변이율로 변이 - 두 유전정보 선택해서 swap.
                #9. 만약 valid한 해면 새로운 해집단에 넣는다.
            #10. 자식 해집단의 fitness 측정
            #11. 부모와 자식의 해집단을 모두 합치고, fitness 상위 40%를 뽑고,
            #   10%는 나머지 60%에서 랜덤으로 선택해 똑같은 크기의 새로운 해집단 구성.(다음 세대)
            # best chromosome 갱신.
            # best fitness 더 안좋아지면 카운트 증가, generation 증가.
        t1 = time.time()
        print('--------------------------------------------------')
        print('GA Search Start')
        print('초기염색체 생성중...')
        print(num_of_chrom, self.count_limit)
        parent_chromosomes, unloaded = self.initial_chromosomes(problem, self.count_limit)
        print('초기염색체 생성완료')
        print('초기염색체 적합도 측정중...')
        parent_fitness = self.measure_fitness(parent_chromosomes, first = True)
        print("적합도: ", parent_fitness)
        init_fitness = max(parent_fitness)

        print('초기염색체 적합도 측정완료')

        best_fitness = max(parent_fitness)
        best_chromosome = parent_chromosomes[parent_fitness.index(best_fitness)]
        print('--------------------------------------------------')
        print(f'현 최적 염색체 : {best_chromosome}, 적합도 : {best_fitness}')
        no_update = 0
        num_of_generations = self.num_of_generation
        generation = 0
        while no_update < self.no_update_count and generation <= num_of_generations:
            generation += 1 
            print(f'{generation}세대 시작')
            print('\t자손 염색체 생성중...')
            children_chromosomes = self.generate_child_chromosomes(parent_chromosomes, parent_fitness, best_chromosome, best_fitness)
            print('\t자손 염색체 생성완료.')
            #print(children_chromosomes)
            print('\t자손 염색체 적합도 측정중...')
            child_fitness = self.measure_fitness(children_chromosomes, args)
            print('자손 염색체 적합도: ', child_fitness)
            print('평균: ', sum(child_fitness) / len(child_fitness))
            print('\t자손 염색체 적합도 측정완료.')
            print('\t다음 세대로 계승될 염색체 선택중...')
            
            whole_chromosomes = parent_chromosomes + children_chromosomes

            whole_fitness = parent_fitness + child_fitness 
            whole = list(zip(whole_chromosomes, whole_fitness))
            whole = sorted(whole, key = lambda x : x[1], reverse = True)
            #print(len(whole))

            num_by_fitness = int(len(parent_chromosomes) * self.gene_select_percent)#다음 세대가 될 염색체들 중 적합도가 높아 선택된 염색체들의 비율
            num_random = len(parent_chromosomes) - num_by_fitness #다음 세대가 될 염색체들 중 랜덤으로 선택된 염색체들의 비율
            new_chromosomes = whole[:num_by_fitness]
            random_idx = random.sample(range(num_by_fitness,len(whole)), num_random)

            
            for idx in random_idx:
                new_chromosomes.append(whole[idx])
            #print(len(new_chromosomes))
            parent_chromosomes = [chro[0] for chro in new_chromosomes] 
            parent_fitness = [chro[1] for chro in new_chromosomes]
            
            print('\t다음 세대 염색체 선택완료')
            #print("선택된 염색체: ", parent_chromosomes)
            max_parent_fitness = whole[0][1]
            max_parent_chrom = whole[0][0]
            print(f'현 최적 적합도 : {best_fitness}, 지역 최적 적합도 : {max_parent_fitness}')

            if best_fitness >= max_parent_fitness:
                no_update += 1 
                
                print(f'\t최적 염색체 갱신 실패! 종료 스택 : {no_update}')
            else:
                no_update = 0 
                best_fitness = max_parent_fitness
                best_chromosome = max_parent_chrom
                print(f'\t최적 염색체 갱신 성공! 종료 스택 : {no_update}')
            print(f'{generation}세대 종료. 다음 세대 시작중...')
            print('--------------------------------------------------')

        best_routes_dict = {car : [] for car in self.problem.cars}
        for idx, car in enumerate(best_chromosome):
            c = self.problem.abst_c[idx]
            best_routes_dict[car].append(c)

        for idx, car in enumerate(best_routes_dict):
            best_routes_dict[car] = self.local_search(best_routes_dict[car])
        print('서치 종료. 최적 솔루션:')
        for car in best_routes_dict:
            print(f'\t {car} route: {best_routes_dict[car]}', end = ', ')
            tot = 0
            car_running_time = self.time_condition(best_routes_dict[car], self.problem.route_distance(best_routes_dict[car]))
            for c in best_routes_dict[car]:
                tot += c.cbm 
            print(f'cap : {car.cap}, cbm = {tot}, time = {car_running_time} / whole_time: {car.whole_travel_time}')
        print(f'초기적합도 : {init_fitness}, 최종적합도 : {best_fitness}')
        print('적재 못한 물품들:')
        for c in unloaded:
            print(f'{c}, cbm : {c.cbm}')
        print(f'총 소요 시간 : {time.time() - t1}')

        return best_routes_dict