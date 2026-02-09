class SimilarityMetric:

    @staticmethod
    def dot(v1, v2):
        result = 0
        for i in range(len(v1)):
            result += v1[i] * v2[i]
        return result

    @staticmethod
    def normalize(vector):
        mag_sq = 0
        for v in vector:
            mag_sq += v * v
        mag = mag_sq**0.5
        if mag == 0:
            return vector
        return [v / mag for v in vector]

    @staticmethod
    def score(metric, query, vector):
        if metric == "dot":
            return SimilarityMetric.dot(query, vector)

        elif metric == "cosine":
            return SimilarityMetric.dot(query, vector)

        elif metric == "l2":
            dist_sq = 0
            for i in range(len(query)):
                diff = query[i] - vector[i]
                dist_sq += diff * diff
            return -dist_sq

        else:
            raise ValueError(f"Unknown metric: {metric}")
