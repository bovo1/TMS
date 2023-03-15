import csv, os 
from Component import * 

class TMSParser:
    def __init__(self, file):
        self.directory = 'data/'
        self.cars_info = 'data/MASTER_CARHANG.XLS'
        self.file = self.directory + file + '/'
        self.hub = file[:4]

    def get_problem(self):
        file_list = os.listdir(self.file) 

        for file in file_list:
            if 'HUB_INFO' in file:
                self.hub_file = self.file + file 
            elif 'CAR_INFO' in file:
                self.car_file = self.file + file 
            elif 'MASTER' in file:
                self.car_info = self.file + file
            elif 'TMS_DETAIL_RESULT' in file:
                self.item_info= self.file + file 
            #elif 'ITEM_INFO' in file:
            #    self.item_info = self.file + file 
        #원본
        #print(self.hub_file,self.car_file,self.car_info,self.item_info)
        #수정
        print("<사용한 파일 경로>")
        print("HUB INFO: ", self.hub_file, '\n', "CAR INFO: ", self.car_file, '\n', "MASTER: ", self.car_info, '\n', "ITEM INFO: ", self.item_info)

        with open(self.item_info, 'r', encoding='CP949', errors='ignore') as f:
            item_info_reader = csv.reader(f, delimiter = '\t', quotechar = '|')
            customers = []
            for index, row in enumerate(item_info_reader):
                if index > 0:
                    #try:
                    if index == 1:
                        hub = Customer(*row)
                        #여기서 hub를 출력하면 왜 0이 나오는지 모르겠다.
                    if '허브' not in row[2]:
                        customers.append(Customer(*row))
                    #except:
                    #    print('Customer Error : ', index, row)

        with open(self.car_file, 'r', encoding='CP949', errors='ignore') as f:
            car_reader = csv.reader(f, delimiter = '\t', quotechar = '|')
            cars = []
            for index, row in enumerate(car_reader):
                if index > 0 :
                    try:
                        car = Car(*row)
                        car.hub = hub
                        cars.append(car)
                    except:
                        print('car Error : ', row)
           
        with open(self.car_info, 'r', encoding='CP949', errors='ignore') as f:
            car_info = csv.reader(f, delimiter = '\t', quotechar = '|')
            for index, row in enumerate(car_info):
                if row[0] == self.hub:
                    district = row[1]
                    car_ids = row[2].replace('"', '').split(',')
                    car_rates = row[3].replace('"', '').split(',')

                    for car in cars:
                        for i in range(len(car_ids)):
                            if car.id == car_ids[i]:
                                try:
                                    car.rate += [int(car_rates[i])]
                                except:
                                    car.rate = [int(car_rates[i])]
                                try:
                                    car.district += [district]
                                except:
                                    car.district = [district]
        print('cars:',cars)
        print('customers:',customers)
        print('hub:',hub)
        problem = Problem(cars, customers, hub)
        return problem