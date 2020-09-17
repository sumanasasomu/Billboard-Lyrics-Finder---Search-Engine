from collections import OrderedDict
from csv import reader
import string #to remove spaces
import re
import binascii
import numpy as np
import sympy
import random
import nltk
import math
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import zlib
import itertools
from operator import itemgetter
# nltk.download('stopwords')

NUM_HASH_FUNCS = 200
NUM_DOCS = 50
INF = 2**32
k = 4
B_BANDS = 40
B_ROWS = 5
threshold = 0

def shingling():
    '''
    #Paramters : None
    #Returns list of all document ids , list of all shingles formed and dictionary of documents as Shingledsets.
    #The function reads the given data set and creates k-shingles of the words and also prints the Document ids.
    '''
    f = open('..\lyrics.csv', 'r', encoding = 'latin-1')
    docIdList = set()
    docsAsShingleSets = {}
    allShingles = []
    l = 0
    print("Reading Docs\n")
    for row in reader(f):
        l += 1
        if l == 1:
            continue

        docId = row[0]
        #print(docId)
        docIdList.add(docId)
        body = row[1]
        body += row[2] #includes title and the lyrics (body of the document)

        stop_words = set(stopwords.words('english'))#Preprocessing of the words and returning tokens.
        word_tokens = word_tokenize(body)
        filtered_doc = ''
        for w in word_tokens:
            if w not in stop_words:
                filtered_doc += w
        body = filtered_doc
        lyrics_nospaces_perdoc  = re.sub('[^A-Za-z]+', '', body) # Remove punctuations, spaces and numbers

        # Keep word shingles
        shinglesInDocWords = set()
        # Keep hashed shingles
        shinglesInDocInts = set()

        # Hashed shingles
        for i in range(len(lyrics_nospaces_perdoc)-k+1):
            shingle = lyrics_nospaces_perdoc[i:i+k]
            if (len(shingle) == k):
                shinglesInDocWords.add(shingle)
                hashed_shingle = binascii.crc32(shingle.encode('ASCII')) & 0xffffffff
                shinglesInDocInts.add(hashed_shingle) # Add the shingle to the list
                if hashed_shingle not in allShingles:
                    allShingles.append(hashed_shingle)
        docsAsShingleSets[docId] = shinglesInDocInts # Getting dictionary of Documents with shingles.

        if l == NUM_DOCS+1:
            break

    f.close() # Closing the file
    return docIdList,docsAsShingleSets,allShingles

def invertedIndexMatrixGenerator(docsAsShingleSets,allShingles):
    '''
    #Paramters: docsAsShingleSets (The dictionary of documents with shingles)
    #           allShingles (all shingles generated till now from the corpus)
    #This function generates the posting list for each shingle in allShingles
    #It returns a dictionary of posting list for each shingle
    '''

    print("Generating Inverted Index\n")
    invertedIndexTable = {}
    allShingles = list(set(allShingles))
    for eachShingle in allShingles:
        postingsList = {}
        for j in docsAsShingleSets:
            if (eachShingle in docsAsShingleSets[j]): # If shingle in present in jth document,j is added to the list
                try:
                    postingsList.add(j)
                except:
                    postingsList = {j}
        invertedIndexTable[eachShingle] = postingsList # Inverted index table for each shingle is made

    return allShingles,invertedIndexTable

def matrixGenerator(allShingles,invertedIndexTable):
    '''
    #Parameters: allShingles (list of all shingles in the corpus)
    #            invertedIndexTable (dictionary with posting list for each shingle)
    #This function indexes the shingles and returns a boolean matrix of shingle versus document
    '''

    index_matrix = {}
    index = 0
    # indexing the shingles
    print("Generating Boolean Matrix\n")
    for shingle in allShingles:
        index_matrix[shingle] = index
        index += 1

    # shingle document matrix
    matrix = np.zeros([len(allShingles),NUM_DOCS],dtype = int)
    for shingle in allShingles:
        postinglist = invertedIndexTable[shingle]
        for d in postinglist:
            matrix[index_matrix[shingle]][int(d)] = 1 # Boolean value true for that document corresponding to a shingle

    return matrix

def pickRandomCoeffs(k,maxval):
    '''
    #Parameters: k (Number of random values)
    #            maxval (maximum value for randint)
    # This function returns k number of unique random values
    '''

    randList = [] # Create a list of 'k' random values.

    while k > 0:
        randIndex = random.randint(0, maxval)  # Get a random shingle ID.

        # Ensure that each random number is unique.
        while randIndex in randList:
        randIndex = random.randint(0, maxval) 
        
        # Add the random number to the list.
        randList.append(randIndex)
        k = k - 1
        
    return randList


def find_sign_matrix(matrix,numOfShingles):
    '''
    #Parameters: matrix (boolean matrix of shingles vs docs)
    #            numOfShingles (total number of shingles in corpus)
    #This function picks two random coefficient values and genrates a hashfunction of form h(x)=(ax+b)%c
    #All values are initialised to infinities and each row is mapped to lowest hash function that has a
    #boolean true for that shingle. This new matrix called sigmatrix is returned.
    # example
    # matrix= [[1, 0, 0, 1],
    #         [0, 0, 1, 0],
    #         [0, 1, 0, 1],
    #         [1, 0, 1, 1],
    #         [0, 0, 1, 0]]
    # coeffA = [1,1]
    # coeffB = [1,3]
    # c = 5
    # required output is [[1, 3, 0, 1], [0, 2, 0, 0]]
    '''

    print("Generating signature Matrix\n")
    c = numOfShingles
    while not sympy.isprime(c):
        c += 1

    coeffA = pickRandomCoeffs(NUM_HASH_FUNCS,numOfShingles-1) # Random coefficient A
    coeffB = pickRandomCoeffs(NUM_HASH_FUNCS,numOfShingles-1) # Random coefficient B

    rows, cols, sigrows = len(matrix), len(matrix[0]), len(coeffA)
    # initialize signature matrix with maxint
    sigmatrix = []
    for i in range(sigrows):
        sigmatrix.append([INF] * cols) # List initialized with INF

    for r in range(rows):
        hashvalue = []
        for h in range(sigrows):
            hashvalue.append((coeffA[h] + coeffB[h]*r) % c) # Hash each row

        # if data != 0 and signature > hash value, replace signature with hash value
        for col in range(cols):
            if matrix[r][col] == 0:
                continue
            for i in range(sigrows):
                if sigmatrix[i][col] > hashvalue[i]:
                    sigmatrix[i][col] = hashvalue[i]
    print("sigmatrix\n")
    # print(sigmatrix)
    return sigmatrix

def getbestb(threshold,NUM_HASH_FUNCS, eps=1e0):
    '''
    #Parameters: threshold (difined threshold)
    #            NUM_HASH_FUNCS (number of hash functions)
    #            eps
    # Returns the best value for b by solving an equation
    '''
    for b in range(1, NUM_HASH_FUNCS+1):
        opt = b*math.log10(b)
        val = -1 * NUM_HASH_FUNCS * math.log10(threshold)
        if opt > val-eps and opt < val+eps:
            print("Using number of bands : %d" % (np.round(b)))
            return np.round(b)

def lsh(B_BANDS,docIdList,sig):
    '''
    #Parameters: B_BANDS (Number of bands in signature matrix)
    #            docIdList (List of document ids)
    #            sig (signature matrix)
    #This function first divides the signature matrix into bands and hashes each column onto buckets.
    #This hashing is called Locality Sensitive Hashing.
    #This function returns the list of document to its hash along with the buckets
    '''

    n = NUM_HASH_FUNCS
    b = B_BANDS
    r = n / b

    d = getbestb(threshold,NUM_HASH_FUNCS)
    # Array of dictionaries, each dictionary is for each band which will hold buckets for hashed vectors in that band
    buckets = np.full(b, {})
    # Mapping from docid to h to find the buckets in which document with docid was hashed
    docth = np.zeros((d, b), dtype=int) #doc to hash
    for i in range(b):
        for j in range(d):
            low = int(i*r) # First row in a band
            high = min(int((i+1)*r),n) # Last row in current band
            l = []
            for x in range(low,high):
                l.append(sig[x][j]) # Append each row into l
            h = int(hash(tuple(l)))%(d+1)
            try:
                buckets[i][h].append(j) # If a bucket corresponds to this hash value append this document into it
            except:
                buckets[i][h] = {j} # Else create a new bucket
            docth[j][i] = h
    # print(docth)
    return docth,buckets

def jacsim(doc1, doc2, docsAsShingleSets,sign_matrix):
    '''
    Jackard similarity
    '''
    doc1 = sign_matrix[:,doc1]
    doc2 = sign_matrix[:,doc2]
    intersection = sum(bool(x) for x in np.logical_and(doc1, doc2))
    return (intersection / len(doc1))


def euclidean_distance(x, y, r=2.0):
    '''
    Euclidean distance
    '''
    A = np.linalg.norm(x)
    B = np.linalg.norm(y)
    x = np.divide(x, A)
    y = np.divide(y, B)
    try:
         return np.power(np.sum(np.power(np.subtract(x, y), r)), 1.0/r)
    except (ValueError,ZeroDivisionError):
         print('Please, enter only even values for "r > 0".')
    except IndexError:
         print('Please, the sets must have the same size.')

def cosine_distance(x,y):
    '''
    Cosine distance
    '''
     prodAB = np.dot(x,y)
     #zeros = np.zeros(len(x))
     A = np.linalg.norm(x)
     B = np.linalg.norm(y)
     return prodAB / (A*B)

def get_similar(dn,docIdList,buckets,docth,docsAsShingleSets):
    '''
    #Parameters: dn (The query document number)
    #            docIdList (List of doc ids)
    #            buckets (List of buckets)
    #            docth (doc to hash list)
    #            docAsShingleSets
    # This function finds similar documents given a query document after hashing and bucketing the query document
    # It also evaluates based on various similarity criterion, namely, Jacard similarity, Euclidean distance
    # and cosine similarity
    # It returns a list of similar documents based on decreasing similarity amount
    '''
    
    if str(dn) not in docIdList:
        raise KeyError('No document with the given name found in the corpus.')

    docid = int(dn)
    # Collection of documents similar to docid
    c = []
    # taking union of all buckets in which docid is present
    for b, h in enumerate(docth[docid]):
        c.extend(buckets[b][h])
    c = set(c)
    print(c)

    # Similar documents
    sim_list = []
    for doc in c:
        if doc == docid:
            continue
        sim = jacsim(docid, doc, docsAsShingleSets,sign_matrix)
        sim_list.append((sim, doc))
    sim_list.sort(reverse=True)
    return sim_list

def get_similarcos(dn,docIdList,buckets,docth,docsAsShingleSets,sign_matrix):
    '''
    Similarity for cosine distance
    '''
    if str(dn) not in docIdList:
        raise KeyError('No document with the given name found in the corpus.')

    docid = int(dn)
    # Collection of documents similar to docid
    c = []
    # taking union of all buckets in which docid is present
    for b, h in enumerate(docth[docid]):
        c.extend(buckets[b][h])
    c = set(c)
    print(c)

    # Similar documents
    sim_list = []
    for doc in c:
        if doc == docid:
            continue
        sim = cosine_distance(sign_matrix[:,dn],
                              sign_matrix[:,doc])
        sim_list.append((sim, doc))
    sim_list.sort(reverse=True)
    return sim_list

if __name__ == '__main__':
    docIdList, docsAsShingleSets, allShingles = shingling()
    numOfShingles = len(allShingles)
    signSize = NUM_DOCS
    allShingles, invertedIndexTable = invertedIndexMatrixGenerator(docsAsShingleSets, allShingles)
    matrix = matrixGenerator(allShingles, invertedIndexTable)
    sign_matrix = find_sign_matrix(matrix,numOfShingles)
    docth,buckets = lsh(B_BANDS,docIdList,sign_matrix)
    inputDocID = input("enter the doc ID you want to know similarities of : ")
    start_time = tm.time()
    sim_docs = get_similar(int(inputDocID),docIdList,buckets,docth,docsAsShingleSets,sign_matrix)

    print("Calculating Jaccard similarities....\n")

    found = 0
    for sim, doc in sim_docs:
        if sim >= threshold:
            found = 1
            print('Document Name: ' + str(doc), 'Similarity: ' + str(sim) + '\n')
    if found == 0:
        print("NO similar docs for the given threshold")

    print("--- %s seconds --- \n\n" % (tm.time() - start_time))    
    start_time = tm.time()    
    sim_docs1 = get_similarcos(int(inputDocID),docIdList,buckets,docth,docsAsShingleSets,sign_matrix)

    print("Calculating Cosine similarities....\n")

    found = 0
    for sim, doc in sim_docs1:
        if sim < threshold:
            found = 1
            print('Document Name: ' + str(doc), 'Similarity: ' + str(sim) + '\n')
    if found == 0:
        print("NO similar docs for the given threshold")

    print("--- %s seconds --- \n\n" % (tm.time() - start_time)) 

    start_time = tm.time()
    #Using Euclidean Distance
    sim_docs2 = get_similareucdis(int(inputDocID),docIdList,buckets,docth,docsAsShingleSets,sign_matrix)
    
    print("Calculating Euclidean similarities....\n")


    found = 0
    t = np.sqrt(2 - 2*threshold)
    for sim, doc in sim_docs2:
        if sim < t:
            found = 1
            print('Document Name: ' + str(doc), 'Similarity: ' + str(sim) + '\n')
    if found == 0:
        print("NO similar docs for the given threshold")   

    print("--- %s seconds ---" % (tm.time() - start_time)) 
