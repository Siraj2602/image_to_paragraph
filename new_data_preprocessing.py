import string
import json
from timeit import default_timer as timer
from keras.preprocessing.text import Tokenizer

start = timer()

# load doc into memory
def to_lines(descriptions):
    all_desc = list()
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc

def create_tokenizer(descriptions):
    lines = to_lines(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text
 
# extract descriptions for images
def load_descriptions(doc,dataset):
    mapping = dict()
    #print(doc)
    # process lines
    doc = doc.split('\n')
    doc.pop(0)
    doc = '\n'.join(doc)
    #doc = ''.join(doc.split('\n').pop(0))
    for line in doc.split('\n'):
        # split line by white space
        tokens = line.split(';')
        if len(line) < 2:
            continue
        # take the first token as the image id, the rest as the description
        image_id , image_desc = tokens[1] , tokens[2:]
        if(int(image_id) in dataset):
        # remove filename from image id
        #image_id = image_id.split('.')[0]
        # convert description tokens back to string
             image_desc = ' '.join(image_desc)
        # create the list if needed
        #if image_id not in mapping:
        #    mapping[image_id] = list()
        # store description
             mapping[image_id] = image_desc.split('.')
             mapping[image_id].pop()
    return mapping
 
def clean_descriptions(descriptions):
    # prepare translation table for removing punctuation
    x = list(string.punctuation)
    x.remove(',')
    x = ''.join(x)
    print(x)
    table = str.maketrans('', '', x)
    for key, desc_list in descriptions.items():
        for i in range(len(desc_list)):
            desc = desc_list[i]
            # tokenize
            desc = desc.split()
            # convert to lower case
            desc = [word.lower() for word in desc]
            # remove punctuation from each token
            desc = [w.translate(table) for w in desc]
            # remove hanging 's' and 'a'
            desc = [word for word in desc if len(word)>1 or (',' in word)]
            # remove tokens with numbers in them
            desc = [word for word in desc if word.isalpha() or (',' in word) ] 
            # store as string
            desc_list[i] =  ' '.join(desc)
 
# convert the loaded descriptions into a vocabulary of words
def to_vocabulary(descriptions):
    # build a list of all description strings
    all_desc = set()
    for key in descriptions.keys():
        [all_desc.update(d.split()) for d in descriptions[key]]
    return all_desc
 
# save descriptions to file, one per line
def save_descriptions(descriptions, filename):
    lines = list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + ' ' + desc)
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()
 
filename = 'convertcsv.csv'
# load descriptions
doc = load_doc(filename)

#file = open('train_split_5000.json')
file = open('test_split_3000.json')
k = file.read()
k = json.loads(k)

# parse descriptions
descriptions = load_descriptions(doc,k)
print('Loaded: %d ' % len(descriptions))
# clean descriptions
clean_descriptions(descriptions)
# summarize vocabulary
# print(descriptions)
vocabulary = to_vocabulary(descriptions)
print('Vocabulary Size: %d' % len(vocabulary))
# save to file
save_descriptions(descriptions, 'descriptions_test.txt')

print(timer() - start)
