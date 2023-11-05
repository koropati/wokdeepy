import numpy as np


def centroid_dengan_anggota_terbanyak(kmeans, data):
    # Melakukan prediksi cluster untuk setiap titik data
    labels = kmeans.predict(data)

    # Menghitung jumlah anggota dalam setiap cluster
    cluster_counts = np.bincount(labels)

    # Menentukan cluster dengan anggota terbanyak
    cluster_terbanyak = np.argmax(cluster_counts)

    # Menentukan centroid dari cluster terbanyak
    centroid_terbanyak = kmeans.cluster_centers_[cluster_terbanyak]

    # Menghitung jarak antara semua titik data dalam cluster terbanyak dengan centroidnya
    distances = np.linalg.norm(data - centroid_terbanyak, axis=1)

    # Mengambil titik data yang paling dekat dengan centroid terbanyak
    anggota_terdekat = data[np.argmin(distances)]

    return centroid_terbanyak, anggota_terdekat
