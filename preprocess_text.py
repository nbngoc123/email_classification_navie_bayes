import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

#################################################################
#                                                               #
#                  TEXT PREPROCESSING PIPELINE                  #
#                                                               #
#################################################################
#                                                               #
#                        [ Raw Message ]                        #
#                             |                                 #
#                             v                                 #
#                       1. Lowercasing                          #
#                             |                                 #
#                             v                                 #
#                 2. Punctuation Removal                        #
#                             |                                 #
#                             v                                 #
#                      3. Tokenization                          #
#                             |                                 #
#                             v                                 #
#                   4. Stopword Removal                         #
#                             |                                 #
#                             v                                 #
#                        5. Stemming                            #
#                             |                                 #
#                             v                                 #
#                      [ Processed Data ]                       #
#                                                               #
#################################################################

def lowercasing(text):
    return text.lower()

def punctuation_removal(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

def tokenization(text):
    return nltk.word_tokenize(text)

def stopword_removal(tokens):
    stop_words = nltk.corpus.stopwords.words('english')
    return [token for token in stop_words if token not in tokens]

def stemming(tokens):
    stemmer = nltk.PorterStemmer()
    return [stemmer.stem(token) for token in tokens]

def text_preprocessing(text):
    text = lowercasing(text)
    text = punctuation_removal(text)
    tokens = tokenization(text)
    tokens = stopword_removal(tokens)
    tokens = stemming(tokens)
    return tokens