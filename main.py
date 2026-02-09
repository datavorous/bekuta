from src.flat import FlatIndex, add, search

index = FlatIndex(dim=3, metric="cosine")

add(index, "vec1", [1, 0, 0])
add(index, "vec2", [0, 1, 0])
add(index, "vec3", [0.7, 0.7, 0])


query = [0.8, 0.6, 0]

results = search(index, query, k=2)

for score_val, id_val in results:
    print(f"id: {id_val}, score: {score_val}")