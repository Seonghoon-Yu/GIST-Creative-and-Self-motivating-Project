from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import pandas as pd
import numpy as np


def get_solved_problem(name='chleodnr3'):
    options = webdriver.ChromeOptions()
    options.add_argument("headless")

    # apt install chromium-chromedriver

    name = name
    driver = webdriver.Chrome(options=options)
    driver.get("https://www.acmicpc.net/user/{}".format(name))
    element = driver.find_element(By.CLASS_NAME, 'problem-list')

    problems = element.text
    problems = problems.split(' ')
    problems = list(map(int, problems))
    return problems


def mapping_problems(problems, reverse=False):
    mapping_dict = pd.read_csv('./data/problem_mapping.csv', header=None, index_col=0, squeeze=True).to_dict()
    mapped_problems = []

    if not reverse:
        for key, value in mapping_dict.items():
            if value in problems:
                mapped_problems.append(key)
    elif reverse:
        for key, value in mapping_dict.items():
            if key in problems:
                mapped_problems.append(value)

    return mapped_problems


def sample_negative(mapped_problems, num_neg=400):
    if len(mapped_problems) >= num_neg:
        raise Exception('Too many solved probles')

    num_neg_ = num_neg - len(mapped_problems)

    # flag = True
    # while flag:
    #     flag=False
    #     neg_sample = list(np.random.randint(0, 14458, size=(num_neg_)))
    #
    #     for problem in mapped_problems:
    #         if problem in neg_sample:
    #             flag = True

    neg_sample = list(np.random.randint(0, 14458, size=(num_neg_)))

    count = 0
    for problem in mapped_problems:
        if problem in neg_sample:
            count += 1
            neg_sample.remove(problem)

    while True:
        add_sample = list(np.random.randint(0, 14458, size=(count)))
        neg_sample = neg_sample + add_sample

        count = 0
        for problem in mapped_problems:
            if problem in neg_sample:
                count += 1
                neg_sample.remove(problem)

        if count == 0:
            break

    problem_input = mapped_problems + neg_sample


    if len(problem_input) != num_neg:
        raise Exception('False problem input in sample negative')


    return problem_input


def get_user_id(name, num_neg):
    user_dict = pd.read_csv('./data/user_id.csv', header=None, index_col=0, squeeze=True).to_dict()

    user_id = False

    for id, nickname in user_dict.items():
        if name == nickname:
            user_id = id
            user_id_aug = [user_id] * num_neg
            return user_id, user_id_aug

    if not user_id:
        raise Exception('Not find User id correspond to user name')



def get_user_id___(name):
    options = webdriver.ChromeOptions()
    options.add_argument("headless")

    # apt install chromium-chromedriver

    name = name
    driver = webdriver.Chrome(options=options)
    print('Searching user id')
    for page in range(1,500):
        print(page)
        driver.get("https://solved.ac/ranking/tier?page={0}".format(page))
        elements = driver.find_elements(By.CLASS_NAME, 'css-1ojb0xa')
        for element in elements:
            element = element.text
            element = element.split()
            if name == element[1]:
                id = element[0]
                return id

    if not id:
        raise Exception("Not find id correspond to name")


def get_user_tier(name='pseudope'):
    options = webdriver.ChromeOptions()
    options.add_argument("headless")

    # apt install chromium-chromedriver

    name = name
    driver = webdriver.Chrome(options=options)
    driver.get("https://www.acmicpc.net/user/{}".format(name))
    element = driver.find_element(By.CLASS_NAME, 'problem-list')

    problems = element.text
    problems = problems.split(' ')
    problems = list(map(int, problems))
    return problems


if __name__ == '__main__':
    # problems = get_solved_problem()
    #
    # mapped_problems = mapping_problems(problems)
    #
    # problem_input, neg_sample = sample_negative(mapped_problems)

    name = 'chleodnr3'
    user_id, user_id_aug = get_user_id(name, 400)
    print(len(user_id))












