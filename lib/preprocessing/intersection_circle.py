import itertools

def count_intersection(circle1, circle2):
    x1, y1, r1 = circle1
    x2, y2, r2 = circle2
    jarak_antara_pusat = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    jarak_radius = r1 + r2
    if jarak_antara_pusat < jarak_radius:
        return min(r1, r2) - (jarak_radius - jarak_antara_pusat)
    else:
        return 0

def find_intersection_circle(circles):
    jumlah_intersections = {}
    for pair in itertools.combinations(circles, 2):
        circle1, circle2 = pair
        intersections = count_intersection(circle1, circle2)
        for circle in pair:
            if tuple(circle) in jumlah_intersections:
                jumlah_intersections[tuple(circle)] += intersections
            else:
                jumlah_intersections[tuple(circle)] = intersections

    titik_terbanyak = max(jumlah_intersections, key=jumlah_intersections.get)
    return list(titik_terbanyak)
