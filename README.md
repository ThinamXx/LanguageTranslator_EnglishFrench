# **English to French Language Translator**

**Objective and Problem Statement**
- Language Translation is a key service needed by the people who are travelling as well as for the people who are settling in a new country. In this Project, I will build the Sequential Model that translates the English sentences to French Sentences. Implementation of Natural Language Processing for converting words or texts into numbers which is then trained using Machine Learning Mode to make the predictions. Neural Machine Translation has been used by Google Translate and it is implemented by billion of users around the world.

**Getting the Data**
- I have manually downloaded the Data. You can access to the Data used in this Project from [Click Here](https://github.com/ThinamXx/LanguageTranslator_EnglishFrench/tree/master/Dataset)

**Exploratory Data Analysis**
- I have performed various Exploratory Data Analysis techniques such as Statistical Exploration, Data Visualization with Plotly and WordCloud and Data Cleaning and Data Preparation process such as Tokenization and Padding and so on.

**English Language or Words**
  - **Snapshot of English Language or Words using Plotly**
  
  ![Image](https://res.cloudinary.com/dge89aqpc/image/upload/v1598843184/A_jjho5n.png)
  
  - **Snapshot of the English Language or Words using WordCloud**
  
  ![Image](https://res.cloudinary.com/dge89aqpc/image/upload/v1598843304/B_cqigbl.png)
  
**French Language or Words**
  - **Snapshot of French Language or Words using Plotly**
  
  ![Image](https://res.cloudinary.com/dge89aqpc/image/upload/v1598843427/C_xnmlix.png)
  
  - **Snapshot of French Language or Words using WordCloud**
  
  ![Image](https://res.cloudinary.com/dge89aqpc/image/upload/v1598843527/D_njyjal.png)
  
**Data Preparation: Tokenization and Padding**

```javascript
def tokenize_padding(x, maxlen):
  tokenizer = Tokenizer(char_level=False)
  tokenizer.fit_on_texts(x)
  sequences = tokenizer.texts_to_sequences(x)
  padded = pad_sequences(sequences, maxlen=maxlen, padding="post")
  return tokenizer, sequences, padded
```

### **Recurrent Neural Network**
- A Recurrent Neural Network or RNN contains a temporal loop in which the hidden layers gives output and feed itself. In RNN, extra dimension is added with time. RNN can recall what happened in the previous timestamp so it will work great on sequence of text. RNN allows us to work on sequence of vectors. So, I will use Recurrent Neural Network for this Project.

**Snapshot of the Model**

```javascript
model = Sequential()
model.add(Embedding(english_vocab, 256, input_length=maxlen_eng, mask_zero=True))
model.add(Bidirectional(GRU(256)))
model.add(RepeatVector(maxlen_fre))
model.add(Bidirectional(GRU(256, return_sequences=True)))
model.add(TimeDistributed(Dense(512, activation="relu")))
model.add(Dropout(0.5))
model.add(TimeDistributed(Dense(french_vocab, activation="softmax")))
model.compile(loss=sparse_categorical_crossentropy,
              optimizer=Adam(learning_rate=0.001),
              metrics=['accuracy'])
model.summary()
```

**Model in Production**
- I have finally implemented the Model to predict the French Language or sentences from English Language or sentences. The Model achieved the accuracy of 84% while training only for 10 epochs. The output of the Trained Model is given as follows:

![Image](https://res.cloudinary.com/dge89aqpc/image/upload/v1598844759/E_uliijs.png)
