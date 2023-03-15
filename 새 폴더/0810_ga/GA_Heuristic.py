from Component import * 
import itertools, random, time
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

    def time_condition(self, item, dist):
        #이 거리는 경우의 수에 따라 임시적이고 가변적인 값이 될 수 있으므로
        #차의 속성이 아닌 지역 변수와 리스트에 보관해주도록 한다.
        dist_time = dist / self.car_speed   #km/h 기준
        item_loading_time = (self.loading_unloading_time / 60 )* len(item.customers_list)
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
        
        self.available_car_list = available_cars
        
        car_cap = [i.cap for i in available_cars]
        
        print(car_prob)
        customers =[c for c in problem.abst_c if c.cbm < max(car_cap)]
        unloaded = [c for c in problem.abst_c if c not in customers]
        chromosomes = []

        while len(chromosomes) < num:
            chrom = []
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

                    if self.no_over_capacity == True:
                        #과적 금지
                        if valid_cars_cap[car_idx] - cust.cbm >= car.cap - (car.cap * self.max_capacity): 
                        #if car.cbm + cust.cbm <= car.cap * self.max_capacity:
                            car.cbm += cust.cbm
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
                        #과적 허용
                        if valid_cars_cap[car_idx] - cust.cbm >= car.cap - (car.cap - self.limit_capacity):
                        #if car.cbm + cust.cbm <= car.cap - self.limit_capacity:
                            car.cbm += cust.cbm
                            #print(f"들어감.{car}에 {cust}가")
                            chrom.append(car)
                            break
                        else:
                            valid_cars.remove(car)                           
                            valid_cars_cap.remove(valid_cars_cap[car_idx])                            
                            valid_owned_cars = [i for i in valid_cars if i.id[0] != 'Y']
                            valid_lent_cars = [i for i in valid_cars if i.id[0] == 'Y']
                            if len(valid_owned_cars) > 0 and len(valid_lent_cars) > 0:
                                valid_cars_prob = ([(1 - self.lent_car_percent)/len(valid_owned_cars) for i in valid_owned_cars] +
                                                [self.lent_car_percent/len(valid_lent_cars) for i in valid_lent_cars])
                            elif len(valid_owned_cars) == 0:
                                valid_cars_prob = [1/len(valid_lent_cars) for i in valid_lent_cars]
                            elif len(valid_lent_cars) == 0:
                                valid_cars_prob = [1/len(valid_owned_cars) for i in valid_owned_cars]
            if len(chrom) == len(customers):
                chromosomes.append(chrom)

        return chromosomes, unloaded

    def measure_fitness(self, chromosomes, first = False):
        customers = self.problem.abst_c
        #거리, 시간, 물량, 교차, 허브반경, 적재 초과, (차와 지점간의 거리 -> 이거를 그냥 거리로 계산하겠음),
        #최대 중량비용 등...
        distances = []
        car_item_list = []
        times = []
        quantities = []
        cross_count = []
        hub_radius = []

        #print("염색체: ",chromosomes)
        for chrom in chromosomes:
            #차 - 고객 구조로 이루어진 루트들 만들기
            dictionary = {car : [] for car in self.available_car_list}
            for i in range(len(chrom)):
                car = chrom[i]
                customer = customers[i]
                dictionary[car].append(customer)
                
            for car in dictionary:
                dictionary[car].insert(0, self.problem.hub)
                car_item_list.append([car, dictionary[car]])
            #print(car_item_list)
            #각 루트 거리 구해서 염색체 총 거리 구하기
            total_distance = 0
            car_time = 0    
            car_times = 0
            total_quantity = 0
            for i in range(len(car_item_list)):
                car = car_item_list[i][0]
                cust = self.local_search(dictionary[car])
                
                if car.id[0] != 'Y':
                    total_distance += self.problem.route_distance(cust)
                else:
                    total_distance += self.lent_car_cost * self.problem.route_distance(cust)
                
                car_time = self.time_condition(cust, self.problem.route_distance(cust))
                car_times += car_time

                


            times.append(car_times)
            distances.append(total_distance)

        #print(dictionary)
        if first == True:
            self.fitness_constant = max(distances) * 5 
        #print("distance: ", distances)
        fitness = [(self.fitness_constant / i)**2 for i in distances]

        return fitness

    def is_chrom_valid(self, chromosome):
        dictionary = {car : [] for car in self.problem.cars}
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
        car1 = np.random.choice(self.problem.cars, size = 1, replace = False)[0]
        i = np.random.choice(range(len(chromosome)), size = 1, replace = False)[0]
        chro[i] = car1
        return chro
                
    def generate_child_chromosomes(self, parent_chromosomes, parent_fitness, best_chromosome, best_fitness):
        one_valid = True 
        two_valid = True

        return child_chromosomes

    #def global_search(self, num_of_chrom, num_of_generations, problem : Problem)->dict[Car: list[Customer]]:
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
        print('GA Search Start')
        print('초기염색체 생성중...')
        parent_chromosomes, unloaded = self.initial_chromosomes(problem, num_of_chrom)
        print("초기 염색체: ")
        #print("parent_chromosomes: ", parent_chromosomes)
        print("unloaded: ", unloaded)
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
        no_update = generation = 0
        while no_update < 10 and generation <= num_of_generations:
            generation += 1 
            print(f'{generation}세대 시작')
            print('\t자손 염색체 생성중...')
            children_chromosomes = self.generate_child_chromosomes(parent_chromosomes, parent_fitness, best_chromosome, best_fitness, args)
            print('\t자손 염색체 생성완료.')
            #print(children_chromosomes)
            print('\t자손 염색체 적합도 측정중...')
            child_fitness = self.measure_fitness(children_chromosomes, args)
            print('\t자손 염색체 적합도 측정완료.')
            print('\t다음 세대로 계승될 염색체 선택중...')

            #변경 후
            """
            whole_chromosomes = parent_chromosomes + children_chromosomes
            whole_fitness = parent_fitness + child_fitness
            whole = list(zip(whole_chromosomes, whole_fitness))
            whole = sorted(whole, key = lambda x : x[1], reverse = True)
            """




            #변경 전
            
            whole_chromosomes = parent_chromosomes + children_chromosomes
            #print(len(parent_chromosomes), len(children_chromosomes), len(whole_chromosomes))   #80 80 160
            #print("whole_chromosomes", whole_chromosomes)

            whole_fitness = parent_fitness + child_fitness 
            whole = list(zip(whole_chromosomes, whole_fitness))
            whole = sorted(whole, key = lambda x : x[1], reverse = True)
            #print(len(whole))

            num_by_fitness = int(len(parent_chromosomes) * 0.8)#다음 세대가 될 염색체들 중 적합도가 높아 선택된 염색체들의 비율
            num_random = len(parent_chromosomes) - num_by_fitness #다음 세대가 될 염색체들 중 랜덤으로 선택된 염색체들의 비율

            #print("num by fitness: ", num_by_fitness)   #72
            #print("num_random: ", num_random)   #8
            new_chromosomes = whole[:num_by_fitness]
            random_idx = random.sample(range(num_by_fitness,len(whole)), num_random)
            #print(len(whole))
            #print("new_chromosomes: ", new_chromosomes)
            #print("random_idx: ", random_idx)
            
            for idx in random_idx:
                new_chromosomes.append(whole[idx])
            #print(len(new_chromosomes))
            parent_chromosomes = [chro[0] for chro in new_chromosomes] 
            parent_fitness = [chro[1] for chro in new_chromosomes]
            
            print('\t다음 세대 염색체 선택완료')
            #print("다음 염색체: ", parent_chromosomes)
            print(len(parent_chromosomes))
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
            for c in best_routes_dict[car]:
                tot += c.cbm 
            print(f'cap : {car.cap}, cbm = {tot}')
        print(f'초기적합도 : {init_fitness}, 최종적합도 : {best_fitness}')
        print('적재 못한 물품들:')
        for c in unloaded:
            print(f'{c}, cbm : {c.cbm}')
        print(f'총 소요 시간 : {time.time() - t1}')

        return best_routes_dict