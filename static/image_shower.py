from collections import defaultdict
import matplotlib.pyplot as plt

with open("DatasaurusDozen.tsv", "r") as f:
    data = list(map(lambda tup: (tup[0], float(tup[1]), float(tup[2])), map(str.split, f.readlines()[1:])))

data_dict = defaultdict(lambda: [])
for img, x, y in data:
    data_dict[img].append((x, y))

fig, axes = plt.subplots(3, 5)
axes = axes.flatten()
for ax, (img, data) in zip(axes, data_dict.items()):
    x, y = [pt[0] for pt in data], [pt[1] for pt in data]
    ax.scatter(x, y)
    ax.set_title(img)
    
plt.show()

