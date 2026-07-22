import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8,5))

ax.plot([0, 1, 2, 3], [0, 2, 1, 4])

ax.set_title("Matplotlib Test")

fig.savefig("test.png", dpi=300)

plt.close(fig)

print("Finished")