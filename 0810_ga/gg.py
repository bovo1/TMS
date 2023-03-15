import gmplot
gmap = gmplot.GoogleMapPlotter(30.3164945, 78.03219179999999, 13)

x = [35.6131603836827, 36.0092232550435, 35.9422173383749, 35.9388621228376]
y = [126.910705150079, 126.667050404963, 126.543220329284, 127.131746882224]
gmap.polygon(x, y, face_color='pink', edge_color='cornflowerblue', edge_width=5)
gmap.draw('map.html')