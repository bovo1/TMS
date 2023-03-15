from haversine import haversine
import copy, random
import numpy as np 
import cv2
import gmplot #구글맵
import webbrowser
import os
class Car:
    def __init__(self, *attri):
        self.id = attri[2] 
        self.cap = float(attri[6].strip())
        self.district = None 
        self.cbm = 0
        self.travel_time = 0
        self.whole_travel_time = int(attri[5].strip()) - int(attri[4].strip())
        self.cross_count = 0
        self.item_cross_count = 0

    def __repr__(self):
        return self.id 

class Customer:
    num = 0
    def __init__(self, *attri):
        self.id = attri[2] 
        self.num = Customer.num
        Customer.num += 1 
        if '허브' in self.id:
            self.num = 0 
        self.cbm = float(attri[10])
        self.lat, self.lon = map(float, attri[28:30])
        self.position = (self.lat, self.lon)
        self.address = attri[15] + attri[16]
        #self.loaded = False

    def __repr__(self):
        return str(self.num)

    def distanceTo(self, other) -> int:
        return haversine(self.position, other.position)

class Problem:
    #def __init__(self, cars : list[Car], customers : list[Customer], hub : Customer):
    def __init__(self, cars : list, customers : list, hub : Customer):
        self.cars = cars 
        self.customers = customers 
        self.hub = hub
        self.abst_c = self.abstract_customers(customers)
        self.distances = self.distance_table(self.abst_c + [hub])

    def __repr__(self):
        return f'P_{self.hub} -> {self.cars} and {self.customers}'

    #def abstract_customers(self, customers)->list[Customer]:
    def abstract_customers(self, customers)->list:
        #def bind_customers(customers : list[Customer]) -> Customer:
        def bind_customers(customers : list) -> Customer:
            total_cbm = 0
            cust = copy.copy(customers[0])
            for c in customers:
                total_cbm += c.cbm 
            cust.cbm = total_cbm 
            cust.customers_list = customers 
            return cust
        
        dictionary = {}
        for c in customers:
            try:
                dictionary[c.position].append(c)
            except:
                dictionary[c.position] = [c]
        binded_customers = []
        for cust in dictionary.values():
            binded_customers.append(bind_customers(cust))
        return binded_customers
        
    #def distance_table(self, customers : list[Customer]):
    def distance_table(self, customers : list):
        distances = {}
        for c1 in customers:
            distance = {}
            for c2 in customers:
                if c1.position != c2.position:
                    distance[c2.position] = c1.distanceTo(c2)
                else:
                    distance[c2.position] = 0
            distances[c1.position] = distance 
        return distances

    #def route_distance(self, customers : list[Customer]) -> int:
    def route_distance(self, customers : list) -> int:
        if len(customers) == 0:
            return 0
        route = [self.hub, *customers, self.hub]
        total = 0
        for a, b in zip(route, route[1:]):
            total += self.distances[a.position][b.position] 
        return total

class Painter:
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)
    def __init__(self, problem : Problem):
        self.problem = problem 
        self.map = np.zeros((1000, 1100, 3), np.uint8)

    #def position_to_map_coords(self, customers : list[Customer]):
    def position_to_map_coords(self, customers : list):
        lats = []
        lons = []
        for cust in customers:
            lats.append(cust.position[0])
            lons.append(cust.position[1])

        lat_min, lat_max = min(lats), max(lats)
        lon_min, lon_max = min(lons), max(lons)
        coords = []
        for cust in customers:
            lat = cust.lat 
            lon = cust.lon 
            y = 100 + ((lat - lat_min) / (lat_max - lat_min)) * 600 
            x = 100 + ((lon - lon_min) / (lon_max - lon_min)) * 900 
            cust.coords = (int(x), int(y))
        return 

    #def draw_points(self, customers: list[Customer]):
    def draw_points(self, customers: list):
        if len(customers) == 0:
            return
        for idx, cust in enumerate(customers):
            if cust.num == 0:
                circle_color = Painter.GREEN 
            else:
                circle_color = Painter.RED
            cv2.circle(self.map, center = cust.coords, radius = 3, color = circle_color, thickness = -1)
        return

    def draw_route(self, points, hub, map, gmap3, car, hub_radius):
        if len(points) == 0:
            return
        route = [hub, *points, hub]
        color = tuple((random.randint(0, 255) for _ in range(3)))
        hub_radius = hub_radius * 1000
        latitude_list = []
        longitude_list = []
        for a, b in list(zip(route, route[1:])):
            hcolor = "#"+''.join([random.choice('ABCDEF0123456789') for i in range(6)])
            if a == route[0]:
                line_color = Painter.GREEN
            else:
                line_color = color 
            cv2.line(map,
                     pt1 = a.coords,
                     pt2 = b.coords,
                     color = line_color,
                     thickness = 1,
                     lineType = cv2.LINE_AA)
            latitude_list.append(a.position[0])
            longitude_list.append(a.position[1])
            gmap3.plot(latitude_list, longitude_list, hcolor, edge_width = 2.5)
            gmap3.text(route[-2].position[0], route[-2].position[1], "CAR ID: {0}".format(car),  color = hcolor, size = 5000)
            #gmap3.scatter(latitude_list[:1],longitude_list[:1], face_alpha = 0.05, size = hub_radius, marker=False) #허브 반경
            gmap3.scatter(latitude_list[:1],longitude_list[:1], '#FF0000',size = 300, marker = False) #허브
            gmap3.scatter(latitude_list,longitude_list, hcolor, size = 100, marker = False ) #배송지
        return latitude_list, longitude_list

    def draw_ga_route(self, hub, ga_route, hub_pos, hub_nm, hub_radius):
        #첫번째고객 위경도
        gmap3 = gmplot.GoogleMapPlotter(hub_pos[0],hub_pos[1], 11)

        for car in ga_route:
            if len(ga_route[car]) == 0:
                continue 
            else:
                map_copy = copy.copy(self.map)
                
                latitude_list, longitude_list = self.draw_route(ga_route[car], hub, map_copy, gmap3, car, hub_radius)
                #cv2.imshow(f'route {car}', map_copy) 
                #cv2.waitKey(0)
                #points, hub, map, gmap3, car
                self.draw_route(ga_route[car], hub, self.map, gmap3, car, hub_radius)


        filepath = 'final_' + hub_nm + '.html'
        gmap3.draw(filepath)
        webbrowser.open(filepath)
        #cv2.imshow(f'final', self.map)
        #cv2.waitKey(0)
