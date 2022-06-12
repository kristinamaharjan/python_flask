import time
from selenium import webdriver
import demoji
import os
import csv
from selenium.webdriver.common.by import By

from selenium import webdriver
from selenium.webdriver.chrome.service import Service

s = Service('D:\\flaskProject1\\flaskProject1\\chromedriver.exe')
driver = webdriver.Chrome(service=s)

fields = ['url', 'review', 'rating']

# chrome driver

# opening the file containing data of URL for extraction
dir = os.getcwd()
fullpath = os.path.join(dir, "pageproducts.txt")
f = open(fullpath, "r")
final_x = f.readlines()
print('All URLs: ')
print(final_x)
maindata = []

# extracting the individual url from the file using the loop
for url in final_x:

    # creating try catch block for error checking
    try:
        # creating a data list
        # using driver to open the individual url
        driver.get(url)
        driver.maximize_window()
        # adding a pause
        time.sleep(3)

        # scrolling
        driver.execute_script("window.scrollTo(0, 600);")
        time.sleep(2)
        driver.execute_script("window.scrollTo(0, 800);")

        paginationExists = len(driver.find_elements(
            By.XPATH, '//*[@id="module_product_review"]/div/div/div[3]/div[2]/div/div/button[last()]')) != 0

        if (paginationExists):
            NO_OF_PAGES = driver.find_element_by_xpath(
                '//*[@id="module_product_review"]/div/div/div[3]/div[2]/div/div/button[last()]').text
            print('no of pages: '+str(NO_OF_PAGES))
            nextPageButton = driver.find_element_by_xpath(
                '//*[@id="module_product_review"]/div/div/div[3]/div[2]/div/button[last()]')
            for pageNo in range(int(NO_OF_PAGES)):
                time.sleep(3)
                # using xpath to get the desired html block i.e getting the class content under class item and extracting its text and adding in list
                for count, x in enumerate(driver.find_elements_by_xpath(
                        '//*[contains(concat( " ", @class, " " ), concat( " ", "item", " " ))]//*[contains(concat( " ", @class, " " ), concat( " ", "content", " " ))]'), 1):
                    data = []
                    texttoadd = x.text

                    # filtering emoji
                    textfilter = demoji.replace(texttoadd, "")

                    # add to main data if length of review is greater than zero
                    if (len(textfilter) > 0):
                        data.append(url.replace('\n', ''))
                        data.append(str(textfilter).replace("\n", ""))
                        # using xpath to select the star image (those with a specific color) to then calculate the count of stars i.e. rating given in the review between 1 and 5
                        noOfStars = len(driver.find_elements_by_xpath(
                            '//*[@id="module_product_review"]/div/div/div[3]/div[1]/div['+str(count)+']/div[1]/div/img[@src=\'//laz-img-cdn.alicdn.com/tfs/TB19ZvEgfDH8KJjy1XcXXcpdXXa-64-64.png\']'))
                        data.append(noOfStars)
                        maindata.append(data)
                nextPageButton.click()
        else:
            for count, x in enumerate(driver.find_elements_by_xpath(
                    '//*[contains(concat( " ", @class, " " ), concat( " ", "item", " " ))]//*[contains(concat( " ", @class, " " ), concat( " ", "content", " " ))]'), 1):
                data = []
                texttoadd = x.text

                # filtering emoji
                textfilter = demoji.replace(texttoadd, "")

                # add to main data if length of review is greater than zero
                if (len(textfilter) > 0):
                    data.append(url.replace('\n', ''))
                    data.append(str(textfilter).replace("\n", ""))
                    # using xpath to select the star image (those with a specific color) to then calculate the count of stars i.e. rating given in the review between 1 and 5
                    noOfStars = len(driver.find_elements_by_xpath(
                        '//*[@id="module_product_review"]/div/div/div[3]/div[1]/div['+str(count)+']/div[1]/div/img[@src=\'//laz-img-cdn.alicdn.com/tfs/TB19ZvEgfDH8KJjy1XcXXcpdXXa-64-64.png\']'))
                    data.append(noOfStars)
                    maindata.append(data)

    except Exception as e:
        print("Error"+str(e))

# writing in csv file
with open('dataWithRatings.csv', 'w', newline='', encoding='utf-8') as f:
    write = csv.writer(f)
    write.writerow(fields)
    write.writerows(maindata)

driver.close()
