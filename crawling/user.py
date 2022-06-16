from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import pandas as pd

options = webdriver.ChromeOptions()
options.add_argument("headless")

id = []

for page in range(451, 501):
    driver = webdriver.Chrome(options=options)
    driver.get("https://solved.ac/ranking/tier?page={0}".format(page))

    for i in range(2, 102):
        name = driver.find_element_by_xpath("//*[@id=\"__next\"]/div[2]/div[2]/div[1]/div/div[{0}]/div[2]/a/b".format(i))
        id.append(name.text)

    driver.quit()

user_df = pd.DataFrame(id, index=range(45000,50000))
user_df.to_csv("./users_df_451-500.csv")

