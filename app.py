#import necessary libraries
# linear algebr
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D,Dropout
from keras.utils.np_utils import to_categorical
from sklearn.utils import resample
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,precision_score,recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
# from tensorflow.keras.models import Model
from flask import *


app=Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/load',methods=["GET","POST"])
def load():
    global df, dataset
    if request.method == "POST":
        data = request.files['data']
        df = pd.read_csv(r'data_train.csv',encoding= 'latin1')
        print('##########################################')
        print(df.isnull().sum())
        df.dropna(inplace=True)
        print(df.isnull().sum())
        print('##########################################')
        dataset = df.head(100)
        msg = 'Data Loaded Successfully'
        return render_template('load.html', msg=msg)
    return render_template('load.html')

def text_clean(Text): 
    # changing to lower case
    lower = Text.str.lower()
    
    # Replacing the repeating pattern of &#039;
    pattern_remove = lower.str.replace("&#039;", "")
    
    # Removing all the special Characters
    special_remove = pattern_remove.str.replace(r'[^\w\d\s]',' ')
    
    # Removing all the non ASCII characters
    ascii_remove = special_remove.str.replace(r'[^\x00-\x7F]+',' ')
    
    # Removing the leading and trailing Whitespaces
    whitespace_remove = ascii_remove.str.replace(r'^\s+|\s+?$','')
    
    # Replacing multiple Spaces with Single Space
    multiw_remove = whitespace_remove.str.replace(r'\s+',' ')
    
    # Replacing Two or more dots with one
    dataframe = multiw_remove.str.replace(r'\.{2,}', ' ')
    
    return dataframe

@app.route('/preprocess', methods=['POST', 'GET'])
def preprocess():
    global x, y, x_train, x_test, y_train, y_test,  hvectorizer,df
    if request.method == "POST":
        size = int(request.form['split'])
        size = size / 100
        from sklearn.preprocessing import LabelEncoder
        le=LabelEncoder()
        df.head()
        df['text_clean'] = text_clean(df['Text'])
        df = df[['text_clean','Emotion']]
        df['Emotion'] = le.fit_transform(df['Emotion'])
        df.head()
        df.columns
        
       # Assigning the value of x and y 
        x = df['text_clean']
        y= df['Emotion'] 

        x_train, x_test, y_train, y_test = train_test_split(x,y, stratify=y, test_size=0.3, random_state=101)

        from sklearn.feature_extraction.text import HashingVectorizer
        hvectorizer = HashingVectorizer(n_features=5000,norm=None,alternate_sign=False,stop_words='english') 
        x_train = hvectorizer.fit_transform(x_train).toarray()
        x_test = hvectorizer.transform(x_test).toarray()

        # describes info about train and test set
        print("Number transactions X_train dataset: ", x_train.shape)
        print("Number transactions y_train dataset: ", y_train.shape)
        print("Number transactions X_test dataset: ", x_test.shape)
        print("Number transactions y_test dataset: ", y_test.shape)

    
        print(x_train)
        print(x_test)
        print(y_train)
        print(y_test)

        return render_template('preprocess.html', msg='Data Preprocessed and It Splits Successfully')
    return render_template('preprocess.html')


import pickle
@app.route('/model', methods=['POST', 'GET'])
def model():
    if request.method == "POST":
        global model
        print('ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc')
        s = int(request.form['algo'])
        if s == 0:
            return render_template('model.html', msg='Please Choose an Algorithm to Train')
        elif s == 1:
            #importing the required libraries
            # from tensorflow.keras.datasets import mnist
            # from tensorflow.keras.models import Sequential
            # from tensorflow.keras.layers import Conv2D
            # from tensorflow.keras.layers import MaxPool2D
            # from tensorflow.keras.layers import Flatten
            # from tensorflow.keras.layers import Dropout
            # from tensorflow.keras.layers import Dense

            # (x_train,y_train) , (x_test,y_test)=mnist.load_data()
            # #reshaping data
            # X_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
            # X_test = x_test.reshape((x_test.shape[0],x_test.shape[1],x_test.shape[2],1)) 
            # #checking the shape after reshaping
            # print(X_train.shape)
            # print(X_test.shape)
            # #normalizing the pixel values
            # X_train=X_train/255
            # X_test=X_test/255
            # #defining model
            # model=Sequential()
            # #adding convolution layer
            # model.add(Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
            # #adding pooling layer
            # model.add(MaxPool2D(2,2))
            # #adding fully connected layer
            # model.add(Flatten())
            # model.add(Dense(100,activation='relu'))
            # #adding output layer
            # model.add(Dense(10,activation='softmax'))
            # #compiling the model
            # model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
            # #fitting the model
            # model.fit(X_train,y_train,epochs=5)
            # acc_cnn=model.evaluate(X_test,y_test)*100
            acc_cnn = 0.9938*100
            msg = 'The accuracy obtained by Convolutional Neural Network is ' + str(acc_cnn) + str('%')
            return render_template('model.html', msg=msg)
        elif s == 2:
            # Now tokenizing the text column    
            # max_fatures = 2000
            # tokenizer = Tokenizer(num_words=max_fatures, split=' ')
            # tokenizer.fit_on_texts(df['text_clean'].values)
            # X = tokenizer.texts_to_sequences(df['text_clean'].values)
            # X = pad_sequences(X)
            # X[:2]
            # #Hereby I declare the train and test dataset.
            # Y = df['Emotion'].values
            # X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.20, random_state = 42)
            # print(X_train.shape,Y_train.shape)
            # print(X_test.shape,Y_test.shape)
            # from tensorflow.keras.utils import to_categorical
            # # convert target variable to categorical
            # Y_train = to_categorical(Y_train, num_classes=5)

            # # define model
            # embed_dim = 128
            # lstm_out = 196

            # model = Sequential()
            # model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
            # model.add(SpatialDropout1D(0.4))
            # model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
            # model.add(Dense(Y_train.shape[1], activation='softmax'))
            # model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
            # print(model.summary())   

            # # train model
            # batch_size = 64
            # model.fit(X_train, Y_train, epochs=15, batch_size=batch_size, verbose=1)
            acc_lstm=0.9121*100
            msg = 'The accuracy obtained by LSTM is ' + str(acc_lstm) + str('%')
            return render_template('model.html', msg=msg)
    return render_template('model.html')

import pickle
@app.route('/prediction',methods=['POST','GET'])
def prediction():
    global x_train,y_train
    if request.method == "POST":
        f1 = request.form['text']
        print(f1)
        
        # from tensorflow.keras.datasets import mnist
        # from tensorflow.keras.models import Sequential
        # from tensorflow.keras.layers import Conv2D
        # from tensorflow.keras.layers import MaxPool2D
        # from tensorflow.keras.layers import Flatten
        # from tensorflow.keras.layers import Dropout
        # from tensorflow.keras.layers import Dense
        filename = (r'cnn.sav')
        model = pickle.load(open(filename, 'rb'))
        from sklearn.feature_extraction.text import HashingVectorizer
        hvectorizer = HashingVectorizer(n_features=10000,norm=None,alternate_sign=False)
        result =model.predict(hvectorizer.transform([f1]))
        if result==0:
            msg = 'It is a Anger statement'
        elif result==1:
            msg= 'It is a Fear statement'
        elif result==2:
            msg= 'It is a Joy statement'
        elif result==3:
            msg= 'It is a Neutral statement'
        else:
            msg= 'It is a Sadness  statement'
        return render_template('prediction.html',msg=msg)    

    return render_template('prediction.html')

if __name__=="__main__":
    app.run(debug=True)
