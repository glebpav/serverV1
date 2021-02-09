from selenium import webdriver


def parseLenta(url):
    print("here")
    driver = webdriver.Chrome()
    driver.get(url)
    allText = driver.find_element_by_class_name("b-topic__content")
    print(allText.text)
    driver.quit()
    return allText
