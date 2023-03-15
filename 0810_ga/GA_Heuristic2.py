from Component import * 
import itertools, random, time
class GA:
    def __init__(self, problem):
        self.problem = problem 

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

    def initial_chromosomes(self, problem, num):
        cars = [i for i in problem.cars]
        owned_cars = [i for i in cars if i.id[0] != 'Y']
        lent_cars = [i for i in cars if i not in owned_cars]
        car_cbm = [i.cap for i in cars]
        car_prob = [0.8/len(owned_cars) for i in owned_cars] + [0.2/len(lent_cars) for i in lent_cars]
        print(car_prob)
        customers =[c for c in problem.abst_c if c.cbm < max(car_cbm)]
        unloaded = [c for c in problem.abst_c if c not in customers]
        chromosomes = []

        #car_total_distance = [0] * len(cars)

        while len(chromosomes) < num:
            chrom = []
            valid_cars = copy.copy(cars)
            valid_cars_cbm = car_cbm[:]
            valid_cars_prob = car_prob[:]
            for cust in customers:
                while True:
                    if len(valid_cars) == 0:
                        print(f'customer {cust} cbm : {cust.cbm}', valid_cars_cbm)
                        break 
                    car = np.random.choice(valid_cars, size = 1, p = valid_cars_prob)[0] 
                    car_idx = valid_cars.index(car)
                    if valid_cars_cbm[car_idx] - cust.cbm > 0 :
                        valid_cars_cbm[car_idx] -= cust.cbm 
                        chrom.append(car)
                        break 
                    else:
                        #valid_cars.remove(car)
                        #valid_cars_cbm.remove(valid_cars_cbm[car_idx])
                        if valid_cars_cbm[car_idx] <= 1:
                            valid_cars.remove(car)
                            valid_cars_cbm.remove(valid_cars_cbm[car_idx])

                        valid_owned_cars = [i for i in valid_cars if i.id[0] != 'Y']
                        valid_lent_cars = [i for i in valid_cars if i.id[0] == 'Y']
                        if len(valid_owned_cars) > 0 and len(valid_lent_cars) > 0:
                            valid_cars_prob = ([0.8/len(valid_owned_cars) for i in valid_owned_cars] +
                                               [0.2/len(valid_lent_cars) for i in valid_lent_cars])
                        elif len(valid_owned_cars) == 0:
                            valid_cars_prob = [1/len(valid_lent_cars) for i in valid_lent_cars]
                        elif len(valid_lent_cars) == 0:
                            valid_cars_prob = [1/len(valid_owned_cars) for i in valid_owned_cars]
            if len(chrom) == len(customers):
                chromosomes.append(chrom)
        #print("chromosomes",chromosomes, len(chromosomes), len(chromosomes[1]), len(chromosomes[5]))
        return chromosomes, unloaded

    def measure_fitness(self, chromosomes, first = False):
        customers = self.problem.abst_c
        distances = []
        #print("염색체: ",chromosomes)
        for chrom in chromosomes:
            #차 - 고객 구조로 이루어진 루트들 만들기
            dictionary = {car : [] for car in self.problem.cars}
            for i in range(len(chrom)):
                car = chrom[i]
                customer = customers[i]
                dictionary[car].append(customer)
            
            #각 루트 거리 구해서 염색체 총 거리 구하기
            total = 0
            for car in dictionary:
                cust = self.local_search(dictionary[car])
                
                if car.id[0] != 'Y':
                    total += self.problem.route_distance(cust)
                else:
                    total += 1.2 * self.problem.route_distance(cust) 
            distances.append(total)
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
        #repeat_num = int(len(parent_chromosomes) * 0.1)
        repeat_num = 3
        total_fitness = sum(parent_fitness)
        #total_fitness = sum(parent_fitness[repeat_num:])
        select_prob = [i/total_fitness for i in parent_fitness]
        #select_prob = [i/total_fitness for i in parent_fitness[repeat_num:]]
        child_chromosomes = []
        cross_rate = 0.7; mutate_rate = 0.01
        chromosomes_similarity = 0


        
        for i in range(repeat_num):
            child_chromosomes.append(best_chromosome)

        while len(child_chromosomes) < len(parent_chromosomes):
            i, j = np.random.choice(range(len(parent_chromosomes)), size = 2, replace = False, p = select_prob)
            #i, j = np.random.choice(range(len(parent_chromosomes[repeat_num:])), size = 2, replace = False, p = select_prob)
            chro1, chro2 = parent_chromosomes[i], parent_chromosomes[j] 
            #2개를 적합도의 비율에 따라 랜덤으로 뽑는다.
            #임의의 2개의 염색체의 적합도를 계산해서 하나 이상의 적합도가 best_fitness보다 낮다면 교체.
            #2개 다 best_fitness 보다 낮다면 제일 낮은 부모 염색체와 교체.
            """
            chro1_fitness = parent_fitness[i]
            chro2_fitness = parent_fitness[j]
            is_now_best_chrom1 = False
            is_now_best_chrom2 = False
            chromosomes_fitness_list = [chro1_fitness, chro2_fitness, best_fitness]
            #오름차순으로 정렬하여 가장 작은 값이 best_fitness가 아니라면 교체해야함.
            
            chromosomes_fitness_list = sorted(chromosomes_fitness_list)
            if chromosomes_fitness_list[0] != best_fitness and chromosomes_fitness_list[0] == chro1_fitness:
                chro1 = best_chromosome
                is_now_best_chrom1 = True
            elif chromosomes_fitness_list[0] != best_fitness and chromosomes_fitness_list[0] == chro2_fitness:
                chro2 = best_chromosome
                is_now_best_chrom2 = True
            """

            #부모 유전정보 유사 정도를 계산함. -> 유사 정도로 초기와 중기, 후기를 구분하여 변이율 조정하기 위함.

            #변경 전
            """
            for i in range(len(chro1)):
                for j in range(len(chro2)):
                    if chro1[i] == chro2[j]:
                        chromosomes_similarity += 1
                        break
            chromosomes_similarity = (chromosomes_similarity*2)/(len(chro1) + len(chro2)) * 100
            #만약 유사 정도가 85% 이상이면 후기로 본다. 그리고 35% 이하면 초기로 본다.
            if chromosomes_similarity >= 85:
                mutate_rate = 0.05
            elif 35 < chromosomes_similarity < 85:
                mutate_rate = 0.15
            elif chromosomes_similarity <= 35:
                mutate_rate = 0.3
            """

            chro1_fit_diff = abs(parent_fitness[i] - best_fitness)
            chro2_fit_diff = abs(parent_fitness[j] - best_fitness)

            if chro1_fit_diff >= 40:
                mutate_rate_1 = 0.3
            elif 10< chro1_fit_diff <40:
                mutate_rate_1 = 0.2
            elif chro1_fit_diff <= 10:
                mutate_rate_1 = 0.05

            if chro2_fit_diff >= 40:
                mutate_rate_2 = 0.3
            elif 10 < chro2_fit_diff < 40:
                mutate_rate_2 = 0.2
            elif chro2_fit_diff <= 10:
                mutate_rate_2 = 0.05

            child1 = []
            child2 = []
            
            
            for k in range(len(chro1)):
                if random.random() > cross_rate:    #cross_rate보다 크면 child1 에 chro1, 작으면 child1 에 chro2 를 넣는다.
                    child1.append(chro1[k])
                    child2.append(chro2[k])
                else:
                    child1.append(chro2[k])
                    child2.append(chro1[k])

            #if random.random() < mutate_rate:         
            #    child1 = self.mutate(child1)
            #else:
            #   child2 = self.mutate(child2)
            if random.random() < mutate_rate_1:
                child1 = self.mutate(child1)
            elif random.random() < mutate_rate_2:
                child2 = self.mutate(child2) 
            else:
                if mutate_rate_1 > mutate_rate_2:
                    child1 = self.mutate(child1)
                else:
                    child2 = self.mutate(child2)

            if len(child_chromosomes) < len(parent_chromosomes):   
                child_chromosomes.append(child1)
            else:
            #if len(child_chromosomes) > len(parent_chromosomes):
                child_chromosomes.append(child2)
            



            #변경 후
            """
            if is_now_best_chrom1 == False and is_now_best_chrom2 == False:
                for k in range(len(chro1)):
                    if random.random() > cross_rate:    #cross_rate보다 크면 child1 에 chro1, 작으면 child1 에 chro2 를 넣는다.
                        child1.append(chro1[k])
                        child2.append(chro2[k])
                    else:
                        child1.append(chro2[k])
                        child2.append(chro1[k])

                if random.random() < mutate_rate:         
                    child1 = self.mutate(child1)
                #if random.random() < mutate_rate:
                else:
                    child2 = self.mutate(child2) 

            elif is_now_best_chrom1 == True and is_now_best_chrom2 == False:
                child1 = chro1
                for k in range(len(chro2)):
                    if random.random() > cross_rate:
                        child2.append(chro2[k])
                    else:
                        child2.append(chro1[k])
                if random.random() < mutate_rate:
                    child2 = self.mutate(child2)

            elif is_now_best_chrom2 == True and is_now_best_chrom1 == False:
                child2 = chro2
                for k in range(len(chro1)):
                    if random.random() > cross_rate:
                        child1.append(chro1[k])
                    else:
                        child1.append(chro2[k])
                if random.random() < mutate_rate:
                    child1 = self.mutate(child1)

            
            if len(child_chromosomes) < len(parent_chromosomes):    
                child_chromosomes.append(child1)
            else:
            #if len(child_chromosomes) > len(parent_chromosomes):
                child_chromosomes.append(child2)
            """

        return child_chromosomes

    def global_search(self, num_of_chrom, num_of_generations, problem : Problem):
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
            children_chromosomes = self.generate_child_chromosomes(parent_chromosomes, parent_fitness, best_chromosome, best_fitness)
            print('\t자손 염색체 생성완료.')
            #print(children_chromosomes)
            print('\t자손 염색체 적합도 측정중...')
            child_fitness = self.measure_fitness(children_chromosomes)
            print('\t자손 염색체 적합도 측정완료.')
            print('\t다음 세대로 계승될 염색체 선택중...')
            
            whole_chromosomes = parent_chromosomes + children_chromosomes
            whole_fitness = parent_fitness + child_fitness 
            whole = list(zip(whole_chromosomes, whole_fitness))
            whole = sorted(whole, key = lambda x : x[1], reverse = True)
            #print(len(whole))

            num_by_fitness = int(len(parent_chromosomes) * 0.8)#다음 세대가 될 염색체들 중 적합도가 높아 선택된 염색체들의 비율
            num_random = len(parent_chromosomes) - num_by_fitness #다음 세대가 될 염색체들 중 랜덤으로 선택된 염색체들의 비율

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