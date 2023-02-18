import sys, os, shutil

model_name = sys.argv[1]
if_sampling = "False" if sys.argv[2] == "0" else "True"
fit = ['all', 'err', 'div']
fitness_func = fit[(int)(sys.argv[3]) - 1]
fitness = sys.argv[3]
print(model_name + "_" + if_sampling + "_" + fitness_func)

if os.path.exists('./error_count.csv'):  # clear ./error_count.csv
    os.remove('./error_count.csv')
shutil.copy('./error_count_null.csv', './error_count.csv')

if os.path.exists('./r_list.csv'):  # clear ./r_list.csv
    os.remove('./r_list.csv')
shutil.copy('./r_list_null.csv', './r_list.csv')

f = open("./generate_carla.py", "r+")
new = []
for line in f:
    new.append(line)
f.close()
new[13] = 'model_name = "' + model_name + '"\n'
f = open("./generate_carla.py", "w+")
for n in new:
    f.write(n)
f.close()

f1 = open("./sampling.py", "r+")
new1 = []
for line in f1:
    new1.append(line)
f1.close()
new1[5] = 'model_name = "' + model_name + '"\n'
f1 = open("./sampling.py", "w+")
for n in new1:
    f1.write(n)
f1.close()

f2 = open("./fuzz_ga.py", "r+")
new2 = []
for line in f2:
    new2.append(line)
f2.close()
new2[12] = 'if_sampling = ' + if_sampling + '\n'
if fitness == "1" and if_sampling == "True":
    new2[13] = 'is_err_collection = 1' + '\n'
else:
    new2[13] = 'is_err_collection = 0' + '\n'
f2 = open("./fuzz_ga.py", "w+")
for n in new2:
    f2.write(n)
f2.close()

if fitness == "1":
    fitness = "r.f[0] += r.f[1] + (r.f[2] - min_val) / minus"
if fitness == "2":
    fitness = "r.f[0] += 0"
elif fitness == "3":
    fitness = "r.f[0] = r.f[1] + (r.f[2] - min_val) / minus"
f3 = open("./ga.py", "r+")
new3 = []
for line in f3:
    new3.append(line)
f3.close()
new3[246] = "\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20" + fitness + '\n'
f3 = open("./ga.py", "w+")
for n in new3:
    f3.write(n)
f3.close()

f4 = open("./rename.py", "r+")
new4 = []
for line in f4:
    new4.append(line)
f4.close()
new4[3] = 'model_name = "' + model_name + '"\n'
new4[4] = 'if_sampling = "' + if_sampling + '"\n'
new4[5] = 'fitness_func = "' + fitness_func + '"\n'
f4 = open("./rename.py", "w+")
for n in new4:
    f4.write(n)
f4.close()

path = '../scenario_runner-0.9.13/_out/'

shutil.rmtree(path + 'center')
os.mkdir(path + 'center')

if os.path.exists(path + 'label_test.csv'):
    os.remove(path + 'label_test.csv')
shutil.copy(path + 'label_test_null.csv', path + 'label_test.csv')
