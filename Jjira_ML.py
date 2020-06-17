import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE

mypath = r"C:\Users\Phil\Desktop\output\jira_issues_15\output"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]


temp  = {'Labels':[],'Values':[]}
df = pd.DataFrame(temp)

text = [] 
count = 0 
for file in onlyfiles:
    text = [] 
    with  open(mypath + '/' + file, "r", encoding="utf-8") as f:
        text = f.read()
        # DataFrame containing each text and each class
        df.loc[count] = [file.rsplit(sep = '_')[0], text.replace('\n','')]
        count = count+1
        
del count, temp, text


#Get the frequency that each class appears in our dataset
Class_freq = df.groupby("Labels").count()
print(Class_freq)
#Get the 3 classes with the most data
max_Classes = Class_freq.nlargest(3,'Values')
print("The 3 classes with the most files")
#print(Class_freq.Data.nlargest(3))

MC = max_Classes.index.tolist()


data = df[df.Labels == MC[0]].append([df[df.Labels == MC[1]]]).append([df[df.Labels == MC[2]]])
data.reset_index(inplace = True)
data.drop(columns = 'index', inplace= True)

vectorizer = CountVectorizer(stop_words = 'english', max_df=0.6, lowercase = True)
features = vectorizer.fit_transform(data.Values)
X= pd.DataFrame(features.toarray(),columns = vectorizer.get_feature_names())
 
y = data.Labels

''' Classification'''
# Split Train-Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


'''  Fit Multinomial Classifier 
After tests on adjusting the classifier best
prediction score we could get is 78% 
without adjusting the features or testing any 
other classifiers.
Classifiers that were tested were GaussianNB and MultinomialNB
'''
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV

parameters = { 'alpha': [1, 2, 5, 10]}
grid_svc = GridSearchCV( MultinomialNB() , parameters , iid =True, cv = 10)
clf = grid_svc.fit(X_train, y_train)

# Predictions and Confusion Matrix Score
predictions = clf.predict(X_test)


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, predictions)
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(clf, X_test,y_test, display_labels=np.unique(y), cmap=plt.cm.Blues, normalize='true')

''' Exporting model and bag of words '''
import joblib
joblib.dump(vectorizer,'C:/Users/Phil/Projects/Jjira/vectorizer.pkl')
joblib.dump(clf,'C:/Users/Phil/Projects/Jjira/classifier.pkl')


''' T-Sne'''
y.replace({MC[0]: 1, MC[1]: 2, MC[2]: 3})
tsne = TSNE(n_components = 2, random_state = 42)
X_embedded = tsne.fit_transform(X)

category_to_color = { 1: 'red', 2:'blue', 3: 'green'}
category_to_label = { 1: MC[0], 2:MC[1], 3: MC[2]}

#.replace({MC[0]: 1, MC[1]: 2, MC[2]: 3}, inplace= True)

colors = y.replace({ 1: 'r', 2: 'b', 3: 'g'})
y_new =y.replace(MC[0],1).replace(MC[1],2).replace(MC[2],3)

fig , ax = plt.subplots()
for category , color in category_to_color.items():
    mask = y_new == category
    ax.scatter(X_embedded[mask,0],X_embedded[mask,1], s = 0.5, color = color , label = category_to_label[category])
    
ax.legend(loc= 'best')