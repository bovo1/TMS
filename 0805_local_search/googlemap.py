import random
import gmplot
import sys

color_group = ['#FF0000','#00FF00','#0000FF']
rnd_color = color_group[random.randrange(0,len(color_group)-1)]
print(rnd_color)
sys.exit()

gmap3 = gmplot.GoogleMapPlotter(30.3164945, 78.03219179999999, 13)

latitude_list = [ 30.3358376, 30.307977, 30.3216419 ]
longitude_list = [ 77.8701919, 78.048457, 78.0413095 ]


# scatter method of map object 
# scatter points on the google map
gmap3.scatter( latitude_list, longitude_list, '#FF0000',size = 400, marker = False )
  
# Plot method Draw a line in
# between given coordinates
gmap3.plot(latitude_list, longitude_list, 'cornflowerblue', edge_width = 2.5)
  
gmap3.draw( "map13.html" )