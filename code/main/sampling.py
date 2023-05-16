import os
from generate_carla import *
from sampling_predict import *
import pandas as pd

model_name = "1"


def sample_simulate(color1, color2, color3, time1, time2, fog, rain, sun, seed_name):
    mutation_name = seed_name.split("_")

    complete_mutation_name = "seed_0_0_" + mutation_name[3]
    parse_path = "../seed_pool/" + complete_mutation_name
    seed_number = mutation_name[3][0]
    DOMTree = xml.dom.minidom.parse(parse_path)

    ele = DOMTree.documentElement
    e = ele.getElementsByTagName("Entities")[0]
    s = ele.getElementsByTagName("Storyboard")[0]

    xml_path = '../seed_pool/' + seed_name
    variable = {"name": "origin & random parameters"}
    rand_para = [color1, color2, color3, time1, time2, fog, rain, sun]
    Simulation(rand_para, variable, e, s)
    writeBack(xml_path, DOMTree)
    os.system("bash ./ga_sim.sh " + seed_name)
    sample_predict = prenum(model_name, seed_number)

    test_dataset_path = '../scenario_runner-0.9.13/_out/label_test.csv'  # clear csv of each seed of sampling
    df = pd.read_csv(test_dataset_path)
    df.head(2)
    df = df.drop(df.index[0:250])
    df.to_csv(test_dataset_path, index=False, sep=',', encoding="utf-8")

    return sample_predict
