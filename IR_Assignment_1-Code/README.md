# IR_Assignment_1
# Billboard-Lyrics-Finder
CS F469 - Information Retrieval | Vector Space Model |

<p><b>Assignment – 1: Domain Specific Search Engine</b></p>
<p><b>Language Used:</b>	Python </p>

<h2>Aim:</h2>
To make a search engine that retrieves relevent documents when a query is entered using the vector space model and "tf-idf" score for ranking.

<h2>Dataset Used:</h2>
The dataset for the search engine is obtained from the Kaggle. It consists of "Song name", "Artist", "year of production" and the "lyrics".  
The dataset is of the format CSV(Column seperated Vectors) with 5 columns - Doc-Id, Song, Artist, Year, Lyrics.

<h2>Working Model:</h2>

1.	The search engine can be started by running app.py.
2.	The entire corpus is preprocessed and the tf-idf Table is produced.
3.  Enter the words or phrases in lyrics you want to search for.
4.	After each search, the top 10 documents that matches with the query are displayed.
5.	The results will be random if the searched words are not present in any document in the corpus.
6.  On clicking the Document, one can view the entire document with details like 'artists' etc.

<h2>Requirements/Installation:</h2>

To run the following code, Flask and nltk have to be readily installed.
<li>	Flask can be installed by following the documentation in the below link. https://linuxize.com/post/how-to-install-flask-on-ubuntu-18-04/</li>
<li>	ntlk can be installed using ‘ntlk.download()’ in a python shell or in the program.</li>

