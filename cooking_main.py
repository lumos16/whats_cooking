import json
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from stemming.porter2 import stem
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.multiclass import OneVsOneClassifier

def preprocess_text(doc, attr):
        text_data = [lemmatizer.lemmatize(word).lower() for word in doc[attr] if word not in stopwords.words('english')]
        return " ".join(text_data)

if __name__ == '__main__':
    lemmatizer = WordNetLemmatizer()
    with open('train.json') as inputFile:
        trainData = json.load(inputFile)
    ingredientsData = [preprocess_text(doc, 'ingredients') for doc in trainData]
    cuisineData = [stem(doc['cuisine'].lower()) for doc in trainData]
    X_train, X_test, y_train, y_test = train_test_split(ingredientsData, cuisineData, test_size=0.3,random_state=105) # 70% training and 30% test
    tfidf = TfidfVectorizer(binary=True)
    trainedData = tfidf.fit_transform(X_train)
    trainedData = trainedData.astype('float16')
    testData = tfidf.transform(X_test)
    testData = testData.astype('float16')
    clf = svm.SVC(C=150, degree=3, tol=0.0001, kernel="rbf", gamma=1)
    model = OneVsOneClassifier(clf, n_jobs=4)
    model.fit(trainedData, y_train)
    y_pred = model.predict(testData)
    print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))