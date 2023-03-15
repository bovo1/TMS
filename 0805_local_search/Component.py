import copy, cv2
import numpy as np
import time 
from haversine import haversine
import random
import gmplot #구글맵
import webbrowser
import itertools
from shapely import *
from shapely.ops import split, linemerge
import copy
import matplotlib.pyplot as plt

all_four_lat_coords = []
all_four_lon_coords = []
all_latitude_list = []
all_longitude_list = []
alllat = []
alllon = []
count = 0
class Car:
    def __init__(self, *attri):
        self.id = attri[2] 
        self.cap = float(attri[6].strip())
        self.customers = []
        self.district = None
        self.real_cbm = 0
        self.hired_car = attri[3]
        self.dist_cost = 0
        self.total_cost = 0
        self.cost_list = []
        self.cross_count = 0
        self.area = None
        self.item_cross_count = 0
        self.whole_travel_time = int(attri[5].strip()) - int(attri[4].strip())    #정해진 출퇴근시간.
        self.travel_time = 0        #차의 운영 시간 체크

    def __repr__(self):
        return self.id 

    def set_hub(self, hub):
        self.hub = hub 

    @property
    def route(self):
        return [self.hub, *self.customers, self.hub]

    @property
    def cbm(self):
        total = 0
        for customer in self.customers:
            total += customer.cbm 
        return total 


class Customer:
    num = 0
    def __init__(self, *attri):
        self.id = attri[2] 
        self.num = Customer.num 
        Customer.num += 1 
        if '허브' in self.id:
            self.num = 0
        #print(len(attri[10]))
        self.cbm = float(attri[10])
        self.lat, self.lon = map(float, attri[28:30])
        self.position = (self.lat, self.lon)
        self.address = attri[15] + attri[16]
        self.loaded = False 
        #self.zip_code = attri[11]

    def __repr__(self):
        return str(self.num)

    def __hash__(self):
        return hash(self.position)

    def distanceTo(self, other) -> int:
        return haversine(self.position, other.position)
        #return ((self.lat-other.lat) ** 2 + (self.lon - other.lon) ** 2 ) ** 0.5


class Problem:
    #def __init__(self, cars : list[Car], customers : list[Customer], hub : Customer):
    def __init__(self, cars : list, customers : list, hub : Customer):
        self.cars = cars
        self.hub = hub 
        self.customers = customers 
        self.all_places = [hub, *customers] 

        self.abst_c = self.abstract_customers(customers)
        #print("self.abst_c: ", self.abst_c, len(self.abst_c))
        self.abst_all = [hub, *self.abst_c]
        #print("self.abst_all: ", self.abst_all, len(self.abst_all))

        self.distances = {}
        for p1 in self.abst_all:
            distance = {}
            for p2 in self.abst_all:
                if not p1.position == p2.position :
                    distance[p2.position] = p1.distanceTo(p2) 
                else:
                    distance[p2.position] = 0
            self.distances[p1.position] = distance 
        
    def __repr__(self):
        return f"problem from {self.hub} with {self.cars}, {self.customers}"

    def bind_cars(self):
        car_dict = {}
        car = [i for i in self.cars if i.district != None]
        for i in car:
            for j in range(len(i.district)):
                dist = i.district[j] 
                rate = i.rate[j] 
                try:
                    car_dict[dist] += [(i, rate)] 
                except:
                    car_dict[dist] = [(i, rate)] 
        return car_dict

    @property 
    def objective_function(self):
        """objective function = sum of all total distances of the cars"""
        total = 0 
        for car in self.cars:
            total += self.total_distance(car.route) 
        return total

    def objective_function2(self, routes):
        total = 0 
        for route in routes:
            total += self.total_distance(route)
        return total

    def total_distance(self, route):
        if len(route) <= 2:
            return 0
        total = 0
        for i in range(len(route) - 1):
            total += self.distances[route[i].position][route[i+1].position] 
        return total

    #def abstract_customers(self, customers) -> list[Customer]:
    def abstract_customers(self, customers) -> list:
        c_dict = {}
        #print("abstract_customer 함수의 customer 개수: ", len(customers))
        #A = []
        #for i in customers:
        #    A.append(i.position)
        #print(A[5][1])
        #print("A안에 들어있는 각 객체들의 position 정보의 개수: ", len(A))
        #print("중복 제거 후의 개수: ", len(set(A)))
        for c in customers:
            try:
                c_dict[c.position].append(c) 
            except:
                c_dict[c.position] = [c]
        #print("c_dict의 개수: ", len(c_dict))
        abstracted_customers = [] 
        for c1 in c_dict:
            abstracted_customers.append(self.abstracted_cust(c_dict[c1]))

        return abstracted_customers
        
    #def abstracted_cust(self, customers : list[Customer]) -> Customer:
    def abstracted_cust(self, customers : list) -> Customer:
        total_cbm = 0 
        abstracted_cust = copy.copy(customers[0])
        for c in customers:
            total_cbm += c.cbm
        abstracted_cust.cbm = total_cbm 
        abstracted_cust.customers_list = customers 
        return abstracted_cust
    

    def evaluate(self, t : int):
        print("Evaluating...")
        print('hub : ')
        print(self.hub)
        print('+++++++++++++++++++++++++++++++++++++++++++++++++')
        print("customers : ")
        print(self.customers)
        print('_________________________________________________')
        print("Total cars : ")
        print(self.cars)
        print('Car Routes : ')
        print('-------------------------------------------------')
        total = 0
        total_customers = []

        for car in self.cars:
            print(f'car {car.id}')
            #try:
            #    print('car.district',car.district)
            #except:
            #    pass 

            customers = []
            c_abst = car.customers
            for c in car.customers:
                customers += c.customers_list
                total_customers += c.customers_list
            car.customers = customers

            if len(car.customers) != 0:
                print(car.route)
                print('cap :', car.cap, 'cbm :', car.cbm, 'distance : ', self.total_distance([self.hub, *c_abst, self.hub]), 'car total cost: ', car.total_cost)
                print('전체 근무 시간: ',car.whole_travel_time ,'총 소요 시간: ', car.travel_time, '초과한 근무 시간: ', (car.travel_time - car.whole_travel_time) if car.travel_time > car.whole_travel_time else 0)
                print('customers number : ', len(car.customers), f'in {len(c_abst)} points')
                print('교차 횟수: ', car.cross_count)
                total += len(car.customers)
            print('----------------------------------------------')
        total_customers = sorted(total_customers, key = lambda x : x.num)
        unloaded = [i for i in self.customers if not i in total_customers]
        abst_unloaded = self.abstract_customers(unloaded)
        
        
        print('total customers :', len(self.customers))
        print('customers in routes : ', total)
        print('customers unloaded : ', len(unloaded))
        print('customers unloaded points : ', len(abst_unloaded), end = ' ')
        for i in range(len(abst_unloaded)):
            print(f'[{abst_unloaded[i].position}, cbm = {round(abst_unloaded[i].cbm,1)}]', end = ' ')
        print(abst_unloaded)
        print('total time spent : ', time.time() - t)
        print('evaluation done')

class Painter:
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)

    def __init__(self):
        self.map = np.zeros((800,1100,3), np.uint8)
    
    def draw(self, problem : Problem, hub_nm):
        global all_four_lat_coords 
        global all_four_lon_coords 
        global all_latitude_list
        global all_longitude_list
        global alllat
        global alllon
        global count
        color_group = ['#FF0000','#00FF00','#0000FF','#ffff00','#000080','#808080','#006400','#8a2be2']
        #def position_to_map_coords(customers : list[Customer]):
        def position_to_map_coords(customers : list):
            lats = []
            lons = []
            for cust in customers:
                lats.append(cust.position[0])
                lons.append(cust.position[1])
            #if len(lats) == 0:
            #    print("위도 경도 정보 값이 추가되지 않았음.")
            #else: print("위도 경도 정보 값이 추가됨: ", lats)

            lat_min, lat_max = min(lats), max(lats) 
            lon_min, lon_max = min(lons), max(lons)

            #print(lat_min, lon_min)

            for cust in customers:
                lat = cust.lat 
                lon = cust.lon 
                #print("반복 lat, lon: ", lat, lon)
                y = 100 + ((lat - lat_min) / (lat_max - lat_min)) * 600 
                x = 100 + ((lon - lon_min) / (lon_max - lon_min)) * 900 
                #print("Y, X: ", y, x)
                cust.coords = (int(x), int(y))
                #print("CUST.COORDS: ",cust.coords) #-> 값이 있는데요... 왜 없다고 하세요?
                #print("cust의 속성 정보: ",dir(cust))
            return 

        #-----------------------------------------------------------------
        #첫번째고객 위경도
        gmap3 = gmplot.GoogleMapPlotter(problem.customers[0].position[0],problem.customers[0].position[1], 11)

        position_to_map_coords(problem.abst_all)
        #position_to_map_coords(problem.customers)
        #이 밑에 인자도 원래 problem.abst_all 이었다.
        for idx, cust in enumerate(problem.abst_all):
            if idx == 0:
                circle_color = Painter.GREEN
            else:
                circle_color = Painter.RED 
            cv2.circle(self.map, center = cust.coords, radius = 3, color = circle_color,thickness = - 1)
            #cv2.putText(self.map, org = cust.coords, text = str(cust.num), 
            #            fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, color = 100, thickness = 1)
        map_for_all = copy.deepcopy(self.map)

        for car in problem.cars:
            if len(car.customers) == 0:
                continue
            color = tuple(map(int, np.random.choice(range(256), size=3)))
            hcolor = "#"+''.join([random.choice('ABCDEF0123456789') for i in range(6)])
            #hcolor = color_group[random.randrange(0,len(color_group)-1)]
            #print('hcolor',hcolor)

            map_copy = copy.deepcopy(self.map)

            latitude_list = []
            longitude_list = []
            for a, b in list(zip(car.route, car.route[1:])):
                #print(a, car.route)
                #print(type(a), type(car.route))
                #print(b.coords)
                #print(dir(b))
                if '허브' in a.id:
                    line_color = Painter.GREEN
                else:
                    line_color = color
                cv2.line(map_copy,
                         pt1 = a.coords,
                         pt2 = b.coords,
                         color = line_color,
                         thickness = 1, 
                         lineType = cv2.LINE_AA)
                cv2.line(map_for_all,
                         pt1 = a.coords,
                         pt2 = b.coords,
                         color = line_color,
                         thickness = 1, 
                         lineType = cv2.LINE_AA)
                latitude_list.append(a.position[0])
                longitude_list.append(a.position[1])
                #if count == 0:
                all_latitude_list.append(a.position[0])
                all_longitude_list.append(a.position[1])
            #count += 1
                #print('coords',latitude_list,longitude_list)
            #cv2.imshow(f'route {car}', map_copy)
            #print('line_color',line_color)
            alllat = copy.deepcopy(all_latitude_list)
            alllon = copy.deepcopy(all_longitude_list)
            print("위도 리스트: ", len(latitude_list), len(all_latitude_list))
            print("경도 리스트: ", len(longitude_list), len(all_longitude_list))
            gmap3.plot(latitude_list, longitude_list, hcolor, edge_width = 2.5)
            
            gmap3.text(car.route[len(car.route)//2].position[0], car.route[len(car.route)//2].position[1], "CAR ID: {0}".format(car),  color = hcolor, size = 5000)
            gmap3.scatter(latitude_list[:1],longitude_list[:1], '#FF0000',size = 300, marker = False ) #허브
            hub_pos = (latitude_list[0], longitude_list[0])
            gmap3.scatter(latitude_list,longitude_list, hcolor,size = 100, marker = False ) #배송지
            dist_list = []
            for i in range(len(problem.abst_c)):
                dist_list.append(haversine((latitude_list[0], longitude_list[0]),(problem.abst_c[i].position), unit='m'))
                #print(haversine((latitude_list[0], longitude_list[0]),(problem.abst_c[i].position), unit='km'))
            dist_avg = sum(dist_list)/len(dist_list)
            #print("DIST_AVG: ", dist_avg)
            gmap3.scatter(latitude_list[:1],longitude_list[:1], size= dist_avg, marker=False, alpha=0.08) #허브 반경
            """
            if car.position_avg != None:
                gmap3.scatter(car.position_avg[:1], car.position_avg[1:2], color = hcolor, size=dist_avg/2, marker=False, alpha=0.3) #차의 반경
                gmap3.text(car.position_avg[0], car.position_avg[1], "CAR ID: {0}".format(car), color = 'black', size = 500)
            """
            #for i in range(len(problem.cars)):          
                #if problem.cars[i].position_avg != None:
                    #print(problem.cars[i].id, problem.cars[i].position_avg, problem.cars[i].position_avg[:1], problem.cars[i].position_avg[1:2], len(problem.cars[i].position_avg))
                #    gmap3.scatter(problem.cars[i].position_avg[:1], problem.cars[i].position_avg[1:2], color = hcolor, size=dist_avg/2, marker=False, alpha=0.3) #차의 반경
                #if len(problem.cars[i].route) >= 3:
                    #gmap3.text(problem.cars[i].route[len(problem.cars[i].route)//2].position[0], problem.cars[i].route[len(problem.cars[i].route)//2].position[1], "CAR ID: {0}".format(problem.cars[i]),  color = 'white', size = 5000)
            for i in range(len(latitude_list)):
                #
                if latitude_list[i] == min(latitude_list):
                    left_coord = (latitude_list[i], longitude_list[i])
                    all_four_lat_coords.append(latitude_list[i])
                    all_four_lon_coords.append(longitude_list[i])
                elif latitude_list[i] == max(latitude_list):
                    right_coord = (latitude_list[i], longitude_list[i])
                    all_four_lat_coords.append(latitude_list[i])
                    all_four_lon_coords.append(longitude_list[i])
                    #
                if longitude_list[i] == min(longitude_list):
                    down_coord = (latitude_list[i], longitude_list[i])
                    all_four_lat_coords.append(latitude_list[i])
                    all_four_lon_coords.append(longitude_list[i])
                elif longitude_list[i] == max(longitude_list):
                    up_coord = (latitude_list[i], longitude_list[i])
                    all_four_lat_coords.append(latitude_list[i])
                    all_four_lon_coords.append(longitude_list[i])
            #일단 최외곽 좌표가 잘잡혔는지 scatter로 확인
            
            
            print("합쳐진 위도좌표:", all_four_lat_coords)
            #gmap3.scatter(all_four_lat_coords, all_four_lon_coords, 'black', size=1000, marker=False)
            box_dict = {}
            box_list = [i.position for i in problem.customers]
            for i in box_list:
                try: box_dict[i] += 1
                except: box_dict[i]=1
            box_dict_list = sorted(box_dict.items(), key=lambda x: (x[0], x[1]))
            #to_go_list_node = [0] + problem.abst_c
            for i in range(len(box_dict_list)):
                for j in range(len(problem.abst_c)):
                    if box_dict_list[i][0] == problem.abst_c[j].position :
                        gmap3.text(box_dict_list[i][0][0], box_dict_list[i][0][1], problem.abst_c[j], color='red', size=250)

            to_del_list = []
            for l in range(len(all_latitude_list)):
                for v in range(len(problem.abst_c)):
                    if all_latitude_list[l] != problem.abst_c[v].position[0] and all_longitude_list[l] != problem.abst_c[v].position[1]:
                        #print("있어요!!!")
                        to_del_list.append((all_latitude_list[l], all_longitude_list[l]))
            
            for tdl in range(len(to_del_list)):
                if to_del_list[tdl] in all_latitude_list:
                    all_latitude_list.remove(to_del_list[tdl])
                    all_longitude_list.remove(to_del_list[tdl])
                        #print(box_dict_list[i][0][0], box_dict_list[i][0][1],problem.abst_c[j])

        #위도 경도의 평균? 중심 좌표

        print("진짜 위도 리스트 개수:", len(all_latitude_list), len(all_longitude_list), len(alllat))
        center_latitude = sum(latitude_list)/len(latitude_list)
        center_longitude = sum(longitude_list)/len(longitude_list)
        #초록색 점은 모든 좌표들의 평균
        gmap3.scatter([center_latitude], [center_longitude], 'green', size = 300, marker = False)
        #후보가 될 수 있는 외곽 좌표들 중에서
        #최외곽의 위, 아래, 오른쪽, 왼쪽 좌표를 걸러내야 한다.
        for i in range(len(all_four_lat_coords)):
            #
            if all_four_lon_coords[i] == min(all_four_lon_coords):
                left_coord = (all_four_lat_coords[i], all_four_lon_coords[i])
            elif all_four_lon_coords[i] == max(all_four_lon_coords):
                right_coord = (all_four_lat_coords[i], all_four_lon_coords[i])
            #
            if all_four_lat_coords[i] == min(all_four_lat_coords):
                down_coord = (all_four_lat_coords[i], all_four_lon_coords[i])
            elif all_four_lat_coords[i] == max(all_four_lat_coords):
                up_coord = (all_four_lat_coords[i], all_four_lon_coords[i])

        #일단 최외곽 좌표가 잘잡혔는지 scatter로 확인
            
        gg_four_lat_coords = [left_coord[0], right_coord[0], down_coord[0], up_coord[0]]
        gg_four_lon_coords = [left_coord[1], right_coord[1], down_coord[1], up_coord[1]]

        #중복이 있는지 확인.
        not_dup_lat_coords = list(set(gg_four_lat_coords))
        not_dup_lon_coords = list(set(gg_four_lon_coords))
        dup_lat_coords = []
        dup_lon_coords = []

        for i in range(len(gg_four_lat_coords)-1):
            for j in range(i+1, len(gg_four_lat_coords)):
                if gg_four_lat_coords[i] == gg_four_lat_coords[j]:
                    dup_lat_coords.append(gg_four_lat_coords[i])
                    dup_lon_coords.append(gg_four_lon_coords[i])

        print("기존 좌표:", gg_four_lat_coords, gg_four_lon_coords)

        print(not_dup_lat_coords, not_dup_lon_coords)
        print("중복인 좌표: ", dup_lat_coords, dup_lon_coords)
        dup_lat_index = [gg_four_lat_coords.index(x) for x in gg_four_lat_coords if x in dup_lat_coords]
        dup_lon_index = [gg_four_lon_coords.index(x) for x in gg_four_lon_coords if x in dup_lon_coords]
        
        dup_lat_index = list(set(dup_lat_index))
        dup_lon_index = list(set(dup_lon_index))
        
        print("중복 좌표의 인덱스: ",dup_lat_index, dup_lon_index)

        def extract_coords(polygon):
            exterior_coords = polygon.exterior.coords[:]
            interior_coords = []
            for interior in polygon.interiors:
                interior_coords += interior.coords[:]
            return exterior_coords, interior_coords        


        #afew = zip(all_latitude_list, all_longitude_list)
        customer_points = MultiPoint([*zip(all_latitude_list, all_longitude_list)])
        #print(len(customer_points.geoms))
        #print(customer_points.geoms[0].coords[0], customer_points.geoms[0].x)
        fig, ax = plt.subplots()
        #plt.title('convex_hull')
        ax.invert_xaxis()
        ax.invert_yaxis()       
        plt.scatter(all_longitude_list, all_latitude_list)
        #-----------------------------------------------------
        points_convex_hull = customer_points.convex_hull
        fig, ax = plt.subplots()
        plt.title('convex_hull')
        exterior, interior = extract_coords(points_convex_hull)
        x,y = zip(*exterior)
        if points_convex_hull.interiors:
            xi, yi = zip(*interior)
            ax.plot(y,x,yi,xi)
            plt.fill(y,x,yi,xi, color = 'lightblue', alpha = 0.3)
        else:
            ax.plot(y,x)
            plt.fill(y,x,color='blue', alpha = 0.3)
        convex_hull_bounds = points_convex_hull.bounds
        min_x = convex_hull_bounds[0]
        min_y = convex_hull_bounds[1]
        max_x = convex_hull_bounds[2]     
        max_y = convex_hull_bounds[3]
        plt.show()

        fig,ax = plt.subplots()
        border_lines = points_convex_hull.boundary
        x_line2 = LineString([(min_x, alllon[alllat.index(min_x)]), hub_pos,(max_x, alllon[alllat.index(max_x)])])
        y_line2 = LineString([(alllat[alllon.index(min_y)], min_y), hub_pos,(alllat[alllon.index(max_y)], max_y)])
        bor_li = [x_line2, y_line2, border_lines]
        xy_line = line_merge(x_line2, y_line2)
        yx_line = line_merge(y_line2, x_line2)
        #print(xy_line)
        #plt.plot(*xy_line.xy)
        #plt.plot(*yx_line.xy)
        unioned = border_lines.union(xy_line)
        unioned = unioned.union(yx_line)
        #print(unioned)
        border_xy_lines = polygonize([unioned])
        #print(border_xy_lines)
        global polys
        polys = MultiPolygon(border_xy_lines)
        print(polys)
        fill_colors = ['olive', 'cornflowerblue', 'salmon', 'orange']
        col_idx = 0
        for l in polys.geoms:
            exterior, interior = extract_coords(l)
            x,y = zip(*exterior)
            if l.interiors:
                xi, yi = zip(*interior)
                ax.plot(y,x,yi,xi, color = fill_colors[col_idx])
                plt.fill(y,x,yi,xi, color = fill_colors[col_idx])
                gmap3.polygon(y,x,yi,xi, face_color = fill_colors[col_idx])
            else:
                ax.plot(y,x, color = fill_colors[col_idx])
                plt.fill(y,x, color = fill_colors[col_idx])
                gmap3.polygon(y,x, face_color = fill_colors[col_idx])
            col_idx += 1
        
        plt.show()
        #-----------------------------------------------------
        not_dup_lat_list = [x for x in all_latitude_list if x not in gg_four_lat_coords]
        not_dup_lon_list = [x for x in all_longitude_list if x not in gg_four_lon_coords]

        new_left_coord = []
        new_right_coord = []
        new_down_coord = []
        new_up_coord = []

        left_coord_ok = False
        right_coord_ok = False
        down_coord_ok = False
        up_coord_ok = False
        to_find_coord_index_list = []
        for i in range(len(dup_lat_index)):
            to_find_coord_index = gg_four_lat_coords.index(gg_four_lat_coords[dup_lat_index[i]], dup_lat_index[i]+1)
            to_find_coord_index_list.append(to_find_coord_index)
            for j in range(len(all_latitude_list)):
                for k in range(len(gg_four_lat_coords)):                                
                    if all_latitude_list[j] not in gg_four_lat_coords and all_longitude_list[j] not in gg_four_lon_coords:
                        if to_find_coord_index == 0 and left_coord_ok == False:
                            #left_coord
                            temp_lon = min(not_dup_lon_list)
                            temp_lat = all_latitude_list[all_longitude_list.index(min(not_dup_lon_list))]
                            new_left_coord.append((temp_lat, temp_lon))
                            left_coord_ok = True
                            not_dup_lat_list.remove(temp_lat)
                            not_dup_lon_list.remove(temp_lon)
                            break
                        elif to_find_coord_index == 1 and right_coord_ok == False:
                            #right_coord
                            temp_lon = max(not_dup_lon_list)
                            temp_lat = all_latitude_list[all_longitude_list.index(max(not_dup_lon_list))]
                            new_right_coord.append((temp_lat, temp_lon))
                            right_coord_ok = True
                            not_dup_lat_list.remove(temp_lat)
                            not_dup_lon_list.remove(temp_lon)
                            break
                        elif to_find_coord_index == 2 and down_coord_ok == False:
                            #down_coord
                            temp_lat = min(not_dup_lat_list)
                            temp_lon = all_longitude_list[all_latitude_list.index(min(not_dup_lat_list))]
                            new_down_coord.append((temp_lat, temp_lon))     
                            down_coord_ok = True
                            not_dup_lat_list.remove(temp_lat)
                            not_dup_lon_list.remove(temp_lon)
                            break   
                        elif to_find_coord_index == 3 and up_coord_ok == False:
                            #up_coord
                            temp_lat = max(not_dup_lat_list)
                            temp_lon = all_longitude_list[all_latitude_list.index(max(not_dup_lat_list))]
                            new_up_coord.append((temp_lat, temp_lon))
                            up_coord_ok = True
                            not_dup_lat_list.remove(temp_lat)
                            not_dup_lon_list.remove(temp_lon)
                            break
                
                if left_coord_ok == True or right_coord_ok == True or down_coord_ok == True or up_coord_ok == True:
                    break
        
        print(new_left_coord, new_right_coord, len(new_down_coord), new_up_coord)

        print("기존 좌표:", gg_four_lat_coords, gg_four_lon_coords)
        to_find_coord_index_list = list(set(to_find_coord_index_list))
        print("바꿀 위치: ", to_find_coord_index_list)

        for i in range(len(to_find_coord_index_list)):
            if to_find_coord_index_list[i] == 0:
                gg_four_lat_coords[0] = new_left_coord[0][0]
                gg_four_lon_coords[0] = new_left_coord[0][1]
            elif to_find_coord_index_list[i] == 1:
                gg_four_lat_coords[1] = new_right_coord[0][0]
                gg_four_lon_coords[1] = new_right_coord[0][1]
            elif to_find_coord_index_list[i] == 2:
                gg_four_lat_coords[2] = new_down_coord[0][0]
                gg_four_lon_coords[2] = new_down_coord[0][1]
            elif to_find_coord_index_list[i] == 3:
                gg_four_lat_coords[3] = new_up_coord[0][0]
                gg_four_lon_coords[3] = new_up_coord[0][1]

        print("새로운 좌표:", gg_four_lat_coords, gg_four_lon_coords)


        xxx = sum(gg_four_lat_coords)/len(gg_four_lat_coords)
        yyy = sum(gg_four_lon_coords)/len(gg_four_lon_coords)

        #노란점은 네 개의 점 좌표들의 평균
        #gmap3.scatter([xxx], [yyy], 'yellow', size = 300, marker = False)
        #빨간점은 사각형의 무게중심
        #gmap3.scatter([center_x], [center_y], 'red', size = 300, marker = False)
        #전체 폴리곤      
        #print("좌표: ", gg_four_lat_coords, gg_four_lon_coords)
        four_coord = [(gg_four_lat_coords[0], gg_four_lon_coords[0]),(gg_four_lat_coords[1], gg_four_lon_coords[1]),(gg_four_lat_coords[2], gg_four_lon_coords[2]),(gg_four_lat_coords[3], gg_four_lon_coords[3])]
        lat = []
        lon = []
        for i, j in four_coord:
            lat.append(i)
            lon.append(j)

        print("정렬 후 좌표: ", four_coord, lat, lon)
        def ccw(a, b, c):
            #a -> (1, 2)
            #return 값이 양수면 반시계 방향, 음수면 시계방향, 0 이면 평행
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

        #시작점은 가장 왼쪽에 있는 좌표로 설정.
        candidate = []
        for val in four_coord:
            if val != left_coord:
                candidate.append(val)

        #순열 이용
        left_coord_start_candidate = list(itertools.permutations(candidate, 3))
        for k in range(len(left_coord_start_candidate)):
            left_coord_start_candidate[k] = list(left_coord_start_candidate[k])
        for k in range(len(left_coord_start_candidate)):
            left_coord_start_candidate[k].insert(0, left_coord)
            left_coord_start_candidate[k].append(left_coord)
        

        path_list = []
        cross_signal = False
        i,j,k = 0,0,0
        for i in range(len(left_coord_start_candidate)):
            for j in range(len(left_coord_start_candidate[i])-1):
                for k in range(3-j):
                    if intersect_line(left_coord_start_candidate[i][j], left_coord_start_candidate[i][j+1], left_coord_start_candidate[i][j+1+k], left_coord_start_candidate[i][j+2+k]) == True:
                        #교차가 되었다면? -> break로 넘어가기.
                        print("i: ",i,"j: ", j, "k: ",k ,"교차")
                        cross_signal = True
                        break
                    else:
                        print("i: ",i,"j: ", j,"k: ",k, "교차아님.")
                        cross_signal = False

                if cross_signal == True:
                    break

            if cross_signal == False:
                path_list = left_coord_start_candidate[i]
                break
                
        print(path_list)
        real_lat = []
        real_lon = []
        for i, j in path_list:
            real_lat.append(i)
            real_lon.append(j)

        filepath = 'final_'+ hub_nm + ".html"
        gmap3.draw(filepath)
        all_four_lat_coords = []
        all_four_lon_coords = []
        count = 0
        all_latitude_list = []
        all_longitude_list = []
        alllat = []
        alllon = []
        polys = None
        webbrowser.open(filepath)
