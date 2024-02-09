import os

dir = "/home/makramchahine/repos/gym-pybullet-drones/gym_pybullet_drones/examples/cl_realgs_settings"

first_colors = []
second_colors = []
# runs = [str(i) for i in range(100)]
runs = []

for folder in sorted(os.listdir(dir)):
    if os.path.isdir(os.path.join(dir, folder)):
        print(folder)
        if int(folder) > 99:
            continue
        runs.append(folder)
        # read color.txt
        with open(os.path.join(dir, folder, "color.txt"), "r") as f:
            color = f.readline().strip()
            first_colors.append(color)
            color = f.readline().strip()
            second_colors.append(color)


print(first_colors)
print(second_colors)
print(runs)

# count "B" in first_colors
print(first_colors.count("B"))
for color in first_colors:
    if color == "B":
        print("B", end=",")
    else:
        print("R", end=",")

print("separate")
for color in second_colors:
    if color == "B":
        print("B", end=",")
    else:
        print("R", end=",")

for run in runs:
    print(run, end=",")