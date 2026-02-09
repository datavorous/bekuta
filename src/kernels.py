

def dot(v1, v2):
    result = 0
    for i in range(len(v1)):
        result += v1[i] * v2[i]
    return result

def score(metric, query, vector):

    if metric == "dot":
        return dot(query, vector)
    
    elif metric == "cosine":
        # we are making an assumption that the vectors are already normalized so we can just do a dot product
        return dot(query, vector)
    
    elif metric == "l2":
        # neg squared l2 distance because we want higher scores to be better
        dist_sq = 0
        for i in range(len(query)):
            diff = query[i] - vector[i]
            dist_sq += diff * diff
        return -dist_sq
    
    else:
        raise ValueError(f"Unknown metric: {metric}")