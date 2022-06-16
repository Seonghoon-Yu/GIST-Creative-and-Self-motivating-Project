from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import pandas as pd

options = webdriver.ChromeOptions()
options.add_argument("headless")

# csv 파일 읽기
users_df = pd.read_csv('./users_df_451-500.csv', index_col='idx')
users_df = users_df.to_numpy().squeeze()

for id in users_df:
    driver = webdriver.Chrome(options=options)
    driver.get("https://solved.ac/search?query=solved_by%3A{0}".format(id))

    # 첫 페이지
    problems = []

    num_problems = driver.find_elements_by_xpath('//*[@id="__next"]/div[2]/div[3]/div[2]/div/div[1]/div/div')

    for i in range(2, len(num_problems) + 1):
        problem = int(driver.find_element_by_xpath('//*[@id="__next"]/div[2]/div[3]/div[2]/div/div[1]/div/div[{0}]/div[1]/span/a/span'.format(i)).text)
        problems.append([id, problem])

    A_ = driver.find_elements_by_xpath('//*[@id="__next"]/div[2]/div[3]/div[2]/div/div[2]/a')

    pages = 1
    if len(A_) == 0:
        pass
    else:
        pages = int(driver.find_element_by_xpath('//*[@id="__next"]/div[2]/div[3]/div[2]/div/div[2]/a[{0}]/div'.format(len(A_))).text)

    driver.close()

    for page in range(2, pages + 1):
        
        driver = webdriver.Chrome(options=options)
        driver.get("https://solved.ac/search?query=solved_by%3A{0}&page={1}".format(id, page))

        num_problems = driver.find_elements_by_xpath('//*[@id="__next"]/div[2]/div[3]/div[2]/div/div[1]/div/div')

        for i in range(2, len(num_problems) + 1):
            problem = int(driver.find_element_by_xpath('//*[@id="__next"]/div[2]/div[3]/div[2]/div/div[1]/div/div[{0}]/div[1]/span/a/span'.format(i)).text)
            problems.append([id, problem])

# index 조정 index = range_parameter
problems_df = pd.DataFrame(problems, columns=['user_id', 'problem_id'])
# csv 파일 이름, save
problems_df.to_csv("./rating_df_451-500.csv")

