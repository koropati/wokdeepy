import random

class KMeans:
    def __init__(self, k, max_iter=100):
        self.k = k
        self.max_iter = max_iter
        self.centroids = []

    def hitung_jarak(self, x1, y1, x2, y2):
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    def inisialisasi_centroid(self, data):
        self.centroids = random.sample(data, self.k)

    def assign_to_centroids(self, data):
        clusters = [[] for _ in range(self.k)]
        for point in data:
            x, y, r = point
            closest_centroid = min(
                self.centroids, key=lambda c: self.hitung_jarak(x, y, c[0], c[1]))
            clusters[self.centroids.index(closest_centroid)].append(point)
        return clusters

    def hitung_centroid_baru(self, cluster):
        if not cluster:
            return None
        x_total = sum(point[0] for point in cluster)
        y_total = sum(point[1] for point in cluster)
        r_total = sum(point[2] for point in cluster)
        x_baru = x_total / len(cluster)
        y_baru = y_total / len(cluster)
        r_baru = r_total / len(cluster)
        return (x_baru, y_baru, r_baru)

    def fit(self, data):
        self.inisialisasi_centroid(data)
        data = [tuple(point) for point in data]  # Konversi data menjadi tuple
        for _ in range(self.max_iter):
            clusters = self.assign_to_centroids(data)
            new_centroids = [self.hitung_centroid_baru(
                cluster) for cluster in clusters]
            if new_centroids == self.centroids:  # Menggunakan .all()
                break
            self.centroids = new_centroids

    def centroid_dengan_anggota_terbanyak(self):
        # Menghitung jumlah anggota pada setiap centroid
        jumlah_anggota = [len(cluster)
                          for cluster in self.assign_to_centroids()]
        # Mendapatkan indeks centroid dengan anggota terbanyak
        indeks_centroid_terbanyak = jumlah_anggota.index(max(jumlah_anggota))
        # Mendapatkan centroid dengan anggota terbanyak
        centroid_terbanyak = self.centroids[indeks_centroid_terbanyak]
        # Menghitung anggota paling dekat dengan centroid terbanyak
        anggota_terdekat = min(self.assign_to_centroids()[indeks_centroid_terbanyak], key=lambda point: self.hitung_jarak(
            point[0], point[1], centroid_terbanyak[0], centroid_terbanyak[1]))
        return centroid_terbanyak, anggota_terdekat

# Contoh penggunaan
data = [[1, 2, 3], [3, 4, 5], [5, 6, 7], [10, 12, 15], [13, 14, 16], [15, 16, 17]]
k = 2

kmeans = KMeans(k)
kmeans.fit(data)

centroid_terbanyak, anggota_terdekat = kmeans.centroid_dengan_anggota_terbanyak()
print("Centroid dengan anggota terbanyak:", centroid_terbanyak)
print("Anggota paling dekat dengan centroid terbanyak:", anggota_terdekat)
