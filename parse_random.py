import matplotlib.pyplot as plt

with open("tune_nyc_filtered1") as f:
    content = f.readlines()

content = [x.strip() for x in content]

ranks = range(len(content))
lrs = []
for line in content:
    vals = line.split()
    lrs.append(float(vals[5]))


plt.figure()
plt.yscale('log')

plt.scatter(ranks, lrs, s=1)

min = min(lrs)
max = max(lrs)
width = max-min

#plt.ylim([min - (0.1*width), max + (0.1*width)])
plt.show(block=True)