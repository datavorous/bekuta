def normalize(vector):
    mag_sq = 0
    for v in vector:
        mag_sq += v * v
    mag = mag_sq**0.5
    if mag == 0:
        return vector
    return [v / mag for v in vector]
