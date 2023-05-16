import shutil

model_name = "1"
model_name="1"
if_sampling="True"
fitness_func="div"
print("************************************************************************************")
print(output)
shutil.copy("./error_count.csv", output)
shutil.move(output, "./ga_output")

entropy = "./entropy_" + model_name + "_" + if_sampling + "_" + fitness_func + ".csv"
shutil.copy("./entropy.txt", entropy)
shutil.move(entropy, "./ga_output/entropy/")

r_list = "./r_list_" + model_name + "_" + if_sampling + "_" + fitness_func + ".csv"
shutil.copy("./r_list.csv", r_list)
shutil.move(r_list, "./ga_output/r_list/")
