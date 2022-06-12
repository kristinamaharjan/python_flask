import time
import os
from selenium import webdriver

#chrome driver
driver = webdriver.Chrome("C:\\Users\\Legion\\PycharmProjects\\python\\chromedriver.exe")


#creating directory to open a file
dir = os.getcwd();
filepath = os.path.join(dir, "pageproducts.txt")
file = open(filepath, "w")

#getting the first URL to fetch products store
url = "https://www.daraz.com.np/aaron-online-hub/?from=wangpu&lang=en&langFlag=en&page=1&pageTypeId=2&q=All-Products"
driver.get(url)

#finding the pagenumbers
pagenumbers = []
for pagenumber in driver.find_elements_by_class_name("ant-pagination-item"):
    pagenumbers.append(pagenumber.text)

#getting the page number
lastitem = pagenumbers[-1]
for i in range(int(lastitem)):
    new_url = "https://www.daraz.com.np/aaron-online-hub/?from=wangpu&lang=en&langFlag=en&page=" + str(i+1) + "&pageTypeId=2&q=All-Products"
    driver.get(new_url)

    #getting the products list from
    link = []
    elem = driver.find_elements_by_xpath('//*[contains(concat( " ", @class, " " ), concat( " ", "c2prKC", " " ))]//a[@href]')
    for e in elem:
        link.append(e.get_attribute("href"))

        #unique link generation
    uniqueLink = []
    for x in link:
        if x not in uniqueLink:
            uniqueLink.append(x)
            print(uniqueLink)

    for x in uniqueLink:
        file.write(x)
        file.write("\n")





