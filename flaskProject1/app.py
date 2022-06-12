from flask import Flask, request, render_template
import time
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
import demoji
import plotly.express as px
import plotly.graph_objects as go


app = Flask(__name__)


fulldata = []
@app.route('/')
def hello_world():  # put application's code here
    return render_template("index.html")


@app.route("/array/", methods=['POST'])
def move_forward():
    print("Here")
    url = request.form['url']
    print(url)

    def crawl(url):

        # chrome driver
        driver = webdriver.Chrome("D:\\flaskProject1\\flaskProject1\\chromedriver.exe")
        maindata = []


        # creating try catch block for error checking
        try:
            # creating a data list
            # using driver to open the individual url
            driver.get(url)

            driver.maximize_window()
            # adding a pause
            time.sleep(3)

            image_link = driver.find_element_by_xpath('//*[ @ id = "module_item_gallery_1"]/*//img').get_attribute(
                "src")
            title = driver.find_element_by_xpath('//*[ @ id = "module_item_gallery_1"]/*//img').get_attribute("alt")
            # scrolling
            driver.execute_script("window.scrollTo(0, 700);")
            time.sleep(5)
            driver.execute_script("window.scrollTo(0, 800);")

            #getting pagination data
            paginationExists = len(driver.find_elements(
                By.XPATH, '//*[@id="module_product_review"]/div/div/div[3]/div[2]/div/div/button[last()]')) != 0

            #checking if there is pagination
            if (paginationExists):
                NO_OF_PAGES = driver.find_element_by_xpath(
                    '//*[@id="module_product_review"]/div/div/div[3]/div[2]/div/div/button[last()]').text
                print('no of pages: ' + str(NO_OF_PAGES))
                nextPageButton = driver.find_element_by_xpath(
                    '//*[@id="module_product_review"]/div/div/div[3]/div[2]/div/button[last()]')
                for pageNo in range(int(NO_OF_PAGES)):
                    time.sleep(5)
                    # using xpath to get the desired html block i.e getting the class content under class item and extracting its text and adding in list
                    for count, x in enumerate(driver.find_elements_by_xpath(
                            '//*[contains(concat( " ", @class, " " ), concat( " ", "item", " " ))]//*[contains(concat( " ", @class, " " ), concat( " ", "content", " " ))]'),
                            1):
                        texttoadd = x.text
                        # filtering emoji
                        textfilter = demoji.replace(texttoadd, "")
                        # add to main data if length of review is greater than zero
                        if (len(textfilter) > 0):
                            maindata.append(str(textfilter).replace("\n", ""))
                        else:
                            break
                    nextPageButton.click()
            else:
                for count, x in enumerate(driver.find_elements_by_xpath(
                        '//*[contains(concat( " ", @class, " " ), concat( " ", "item", " " ))]//*[contains(concat( " ", @class, " " ), concat( " ", "content", " " ))]'),
                        1):
                    texttoadd = x.text
                    # filtering emoji
                    textfilter = demoji.replace(texttoadd, "")
                    # add to main data if length of review is greater than zero
                    if (len(textfilter) > 0):
                        maindata.append(str(textfilter).replace("\n", ""))
                    else:
                        break

        except Exception as e:
            print("Error" + str(e))
        driver.close()
        myobj = {
            "data":maindata,
            "title":title,
            "image":image_link,
        }
        return myobj



    nltk.download('stopwords')
    nltk.download('punkt')

    #opening testing data using pandas
    data = pd.read_csv('dataWithRatings2.csv', header=0)
    data = data.dropna()

    #seperating data with positive and negative reviews
    positive_reviews = data.loc[data['label'] == 1.0, ['review']]
    negative_reviews = data.loc[data['label'] != 1.0, ['review']]
    positive_reviews = positive_reviews['review'].to_list()
    negative_reviews = negative_reviews['review'].to_list()

    #getting training data
    train_x = positive_reviews + negative_reviews
    #getting training labels
    train_y = np.append(np.ones((len(positive_reviews), 1)), np.zeros((len(negative_reviews), 1)), axis=0)

    #function to clearing the data

    def process_review(review):

        # Remove hyperlinks
        review = re.sub(r'https?:\/\/.*[\r\n]*', '', review)

        # Remove hashtags
        # Only removing the hash # sign from the word
        review = re.sub(r'#', '', review)

        # tokenize reviews
        review_tokens = word_tokenize(review)

        # Import the english stop words list from NLTK
        stopwords_english = stopwords.words('english')

        # Creating a list of words without stopwords
        clean_review = []
        for word in review_tokens:
            if word not in stopwords_english and word not in string.punctuation:
                clean_review.append(word)

        # Instantiate stemming class
        stemmer = PorterStemmer()

        # Creating a list of stems of words in review
        reviews_stem = []
        for word in clean_review:
            stem_word = stemmer.stem(word)
            reviews_stem.append(stem_word)

        return reviews_stem

########################### TRAINING ###################################
    def build_freqs(reviews, yvalues):

        yvalueslist = np.squeeze(yvalues).tolist()

        freqs = {}
        # for y, review in zip(yvalueslist, reviews):
        #     for word in process_review(review):
        #         pair = (word, y)
        #         freqs[pair] = freqs.get(pair, 0) + 1
        #
        # return freqs
        for y, tweet in zip(yvalueslist, reviews):
            for word in process_review(tweet):
                pair = (word, y)
                if pair in freqs:
                    freqs[pair] += 1
                else:
                    freqs[pair] = 1


        return freqs

    def sigmoid(z):
        '''
        Input:
            z: is the input (can be a scalar or an array)
        Output:
            h: the sigmoid of z
        '''
        # calculate the sigmoid of z
        h = 1 / (1 + np.exp(-z))

        return h

    def gradientDescent(x, y, theta, alpha, num_iters):

        '''
        Input:
            x: matrix of features which is (m,n+1)
            y: corresponding labels of the input matrix x, dimensions (m,1)
            theta: weight vector of dimension (n+1,1)
            alpha: learning rate
            num_iters: number of iterations you want to train your model for
        Output:
            J: the final cost
            theta: your final weight vector
        Hint: you might want to print the cost to make sure that it is going down.
        '''

        m = len(x)

        for i in range(0, num_iters):
            # get z, the dot product of x and theta
            z = np.dot(x, theta)

            # get the sigmoid of z
            h = sigmoid(z)

            # calculate the cost function
            J = (-1 / m) * (np.dot(y.T, np.log(h)) + np.dot((1 - y).T, np.log(1 - h)))

            # update the weights theta
            theta = theta - (alpha / m) * np.dot(x.T, h - y)

        J = float(J)
        return J, theta

    def extract_features(review, freqs):

        '''
        Input:
            review  : a list of words for one tweet
            freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
        Output:
            x: a feature vector of dimension (1,3)
        '''
        # process_review tokenizes, stems, and removes stopwords
        word_l = process_review(review)

        # 3 elements in the form of a 1 x 3 vector
        x = np.zeros((1, 3))

        # bias term is set to 1
        x[0, 0] = 1

        # loop through each word in the list of words
        for word in word_l:
            # increment the word count for the positive label 1
            x[0, 1] += freqs.get((word, 1), 0)

            # increment the word count for the negative label 0
            x[0, 2] += freqs.get((word, 0), 0)

        assert (x.shape == (1, 3))
        return x

    def train_naive_bayes(freqs, train_x, train_y):
        '''
        Input:
            freqs: dictionary from (word, label) to how often the word appears
            train_x: a list of tweets
            train_y: a list of labels correponding to the tweets (0,1)
        Output:
            logprior: the log prior. (equation 3 above)
            loglikelihood: the log likelihood of you Naive bayes equation. (equation 6 above)
        '''
        loglikelihood = {}
        logprior = 0
        # calculate V, the number of unique words in the vocabulary
        vocab = set([pair[0] for pair in freqs.keys()])
        V = len(vocab)
        # calculate N_pos and N_neg
        N_pos = N_neg = 0
        for pair in freqs.keys():
            # if the label is positive (greater than zero)
            if pair[1] > 0:
                # Increment the number of positive words by the count for this (word, label) pair
                N_pos += freqs.get(pair, 1)
            # else, the label is negative
            else:
                # increment the number of negative words by the count for this (word,label) pair
                N_neg += freqs.get(pair, 1)
        # Calculate D, the number of documents
        D = len(train_y)
        # Calculate D_pos, the number of positive documents (*hint: use sum(<np_array>))
        D_pos = sum(train_y)
        # Calculate D_neg, the number of negative documents (*hint: compute using D and D_pos)
        D_neg = D - D_pos
        # Calculate logprior
        logprior = np.log(D_pos) - np.log(D_neg)
        # For each word in the vocabulary...
        for word in vocab:
            # get the positive and negative frequency of the word
            freq_pos = freqs.get((word, 1), 0)
            freq_neg = freqs.get((word, 0), 0)
            # calculate the probability that each word is positive, and negative
            p_w_pos = (freq_pos + 1) / (N_pos + V)
            p_w_neg = (freq_neg + 1) / (N_neg + V)
            # calculate the log likelihood of the word
            loglikelihood[word] = np.log(p_w_pos / p_w_neg)
        return logprior, loglikelihood

###############################TESTING and PREDICTION #############################
    def naive_bayes_predict(tweet, logprior, loglikelihood):
        '''
        Input:
            tweet: a string
            logprior: a number
            loglikelihood: a dictionary of words mapping to numbers
        Output:
            p: the sum of all the logliklihoods of each word in the tweet (if found in the dictionary) + logprior (a number)'''
        # process the tweet to get a list of words
        word_l = process_review(tweet)  # initialize probability to zero
        p = 0  # add the logprior
        p += logprior
        for word in word_l:  # check if the word exists in the loglikelihood dictionary
            if word in loglikelihood:
                # add the log likelihood of that word to the probability
                p += loglikelihood[word]
        return p

    def test_naive_bayes(test_x, logprior, loglikelihood):
        """
        Input:
            test_x: A list of tweets
            test_y: the corresponding labels for the list of tweets
            logprior: the logprior
            loglikelihood: a dictionary with the loglikelihoods for each word
        Output:
            accuracy: (# of tweets classified correctly)/(total # of tweets)
        """
        labelPredictions = []
        y_hats = []
        y = 0
        for tweet in test_x:
            data = []
            data.append(tweet)
            prediction = naive_bayes_predict(tweet, logprior, loglikelihood)
            print(tweet+" prediction= "+str(prediction))
            if prediction > 0:
                data.append(1)
            else:
                data.append(0)
            labelPredictions.append(data)
        return labelPredictions

    myobjs = crawl(url)
    testData = myobjs.get("data")
    image = myobjs.get("image")
    title = myobjs.get("title")


    freqs = build_freqs(train_x, train_y)
    X = np.zeros((len(train_x), 3))
    for i in range(len(train_x)):
        X[i, :] = extract_features(train_x[i], freqs)
    # training labels corresponding to X
    Y = train_y
    # Apply gradient descent
    logprior, loglikelihood = train_naive_bayes(freqs, train_x, train_y)
    naiveBayesOutputs = test_naive_bayes(testData, logprior, loglikelihood)
    print('Predictions using naive bayes: ')
    print(naiveBayesOutputs)
    positivecount = 0
    negativecount = 0
    for q in naiveBayesOutputs:
        if q[1] == 0:
            positivecount = positivecount + 1
        elif q[1] == 1:
            negativecount = negativecount + 1

    values = [positivecount,negativecount]
    print("==============================")

    print(values)

    print(image)
    print(title)
    return render_template('result.html', mylist=naiveBayesOutputs, url=url, title=title, img=image, fulldata=values)

@app.route("/graph/", methods=['GET'])
def generate_graph():
    q = request.args.get("q")
    print(q)
    values = list(q.split(","))
    print(values)
    positive = values[0][1]
    negative = values[1][1]


    print(positive)
    print(negative)
    import plotly.graph_objects as go

    labels = ['postive','negative']
    values = [int(positive),int(negative)]
    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
    fig.show()

    return ""


if __name__ == '__main__':
    app.run()
