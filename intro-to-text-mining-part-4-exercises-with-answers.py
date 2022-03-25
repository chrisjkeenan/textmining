#!/usr/bin/env python
# coding: utf-8

# # Intro to text mining - Part - 4 - Exercises with answers

# ## Exercise 1

# #### Task 1
# ##### Load libraries that are used in this module.
# 

# #### Answer:

# In[4]:


get_ipython().system('pip install pyvis')

# Helper packages.
from pathlib import Path
import os 
import pickle
import pandas as pd
import numpy as np
# Cosine similarity and clustering packages.
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import ward, dendrogram, fcluster
from gensim import matutils
# Network creation and visualization.
import networkx as nx
from pyvis.network import Network
# Other plotting tools.
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt


# #### Task 2 
# ##### Set `main_dir` to the location of your `skillsoft` folder (for Mac/Linux/Windows).
# ##### Make `data_dir` from the `main_dir` and concatenate remainder of the path to data directory.

# #### Result:

# In[ ]:


# Set 'main_dir' to location of the project folder
home_dir = Path(".").resolve()
main_dir = home_dir.parent
#main_dir = home_dir
data_dir = str(main_dir) + "/data"
plot_dir = str(main_dir) + "/plots"


# #### Task 3 
# ##### Set the working directory to `data_dir`.
# ##### Check if the working directory is updated to `data_dir`.

# #### Result :

# In[ ]:


# Change the working directory.
os.chdir(data_dir)
# Check the working directory.
print(os.getcwd())


# #### Task 4 
# ##### Load the pickled file from the previous exercises: `titles_clean.sav`, `bow_corpus_ex.sav`,
# ##### `corpus_tfidf_ex.sav`, `dictionary_ex.sav`, `ex_DTM.sav`, `ex_word_counts_array` and `doc_topic_df_ex`
# ##### Save them as `processed_docs_ex`, `bow_corpus_ex`, `corpus_tfidf_ex`, `dictionary_ex`, `DTM_ex` , `ex_word_counts_array` and `doc_topic_df_ex` respectively.
# ##### Load the dataset UN_agreement titles and save it as `UN`.

# #### Result:

# In[ ]:


processed_docs_ex = pickle.load(open("titles_clean.sav","rb"))
bow_corpus_ex = pickle.load(open("bow_corpus_ex.sav","rb"))
corpus_tfidf_ex = pickle.load(open("corpus_tfidf_ex.sav","rb"))
dictionary_ex = pickle.load(open("dictionary_ex.sav","rb"))
DTM_ex = pickle.load(open('ex_DTM.sav', 'rb'))
ex_word_counts_array = pickle.load(open("ex_word_counts_array.sav","rb"))
doc_topic_df_ex = pickle.load(open("doc_topic_df_ex.sav","rb"))

UN = pd.read_csv('UN_agreement_titles.csv')


# #### Task 5 
# ##### Generate a TDM from corpus weighted with TF-IDF, name it `TDM_tf_idf_ex`
# ##### Check the dimensions of the type of the TDM.
# ##### How many terms and documents are there in the 2D array?

# #### Result:

# In[ ]:


# Convert corpus weighted with TF-IDF to a TDM matrix.
TDM_tf_idf_ex = matutils.corpus2dense(corpus_tfidf_ex,
                                      DTM_ex.shape[1],
                                      DTM_ex.shape[0])


print(type(TDM_tf_idf_ex))
print(TDM_tf_idf_ex.shape)


# #### Task 6
# ##### Convert the above TDM into a DTM called `DTM_tf_idf_ex`.
# ##### Print the dimensions of the matrix.
# ##### Save the matrix created as a dataframe called `DTM_df_ex`.
# 
# #### Result:

# In[ ]:


# Transpose matrix to get the DTM.
DTM_tf_idf_ex = TDM_tf_idf_ex.transpose()

print(DTM_tf_idf_ex.shape)


# In[ ]:


# Create the DTM weighted with TF-IDF.
valid_snippets_ex = np.where(ex_word_counts_array >= 3)[0]
print(len(valid_snippets_ex))

DTM_df_ex = pd.DataFrame(DTM_tf_idf_ex,
                         columns = DTM_ex.columns, 
                         index = valid_snippets_ex) #<- set index to original article index
print(DTM_df_ex.head())


# #### Task 7 
# ##### Compute cosine similarity for the `DTM_tf_idf_ex` matrix.
# ##### Print the shape of the matrix.
# ##### Save the similarity matrix as a dataframe called `similarity_df_ex`.
# 
# #### Result:

# In[ ]:


# Compute similarity matrix (a numpy 2D array).
similarity_ex = cosine_similarity(DTM_tf_idf_ex)
print(type(similarity_ex))

print(similarity_ex.shape)

# Create similarity dataframe with appropriate column names and indices.
similarity_df_ex = pd.DataFrame(similarity_ex,
                                columns = valid_snippets_ex,
                                index = valid_snippets_ex)


# ## Exercise 2

# #### Task 1
# ##### Compute a graph from similarity object `similarity_df_ex`.
# ##### Convert the graph into a dataframe in the form of a edgelist called `edgelist_df_ex`.
# ##### Print the shape of `edgelist_df_ex`.

# #### Result :

# In[ ]:


# Create a graph object from the similarity matrix.
graph = nx.from_pandas_adjacency(similarity_df_ex)

# Convert it to a dataframe in a form of an edgelist.
edgelist_df_ex = nx.to_pandas_edgelist(graph)

# Take a look at the data frame of edges.
print(edgelist_df_ex.head())


# In[ ]:


print(edgelist_df_ex.shape)


# #### Task 2
# ##### Create a cosine similarity score distribution by plotting the weights of edges .
# ##### Filter out all pairs of documents with weights below 0.4 and above 0.8.
# ##### Print the head and shape of the new `edgelist_df_ex`.

# #### Result:

# In[ ]:


# Answer:
# Plot the weights of edges (i.e. similarity scores).
plt.hist(edgelist_df_ex['weight'])
plt.xlabel('Cosine similarity score')
plt.title('Cosine similarity score distribution')
plt.show()


# In[ ]:


# Filter out all entries below 0.4 and above 0.8.
edgelist_df_ex = edgelist_df_ex.query('weight>0.4 and weight<0.8')

# Take a look at the dataframe of edges.
print(edgelist_df_ex.head())


# In[ ]:


print(edgelist_df_ex.shape)


# #### Task 3 
# ##### Create an empty network object `network_ex` with the following base parameters:
#     - height - 100%
#     - width - 60%
#     - bgcolor - FFFFF
#     - font_color - 000000

# #### Result:

# In[ ]:


# Create an empty network object.
network_ex = Network(height = "100%",
                     width = "60%",
                     bgcolor = "#FFFFFF",
                     font_color = "#000000")

# Set the physics layout of the network.
network_ex.force_atlas_2based()
network_ex.set_edge_smooth('dynamic')
print(network_ex)


# #### Task 4
# ##### Populate the empty network with edge and node data. Use `edgelift_df_ex` and 
# ##### zip the three necessary columns source, target, and weight into an iterable set of tuples.
# ##### Print network nodes and network edges of your choice.

# #### Result:

# In[ ]:


# Zip columns of edgelist data into a set of tuples.
edge_data = zip(edgelist_df_ex['source'], edgelist_df_ex['target'], edgelist_df_ex['weight'])
# Iterate through the edge data.
for e in edge_data:
    src = e[0] #<- get the source node
    dst = e[1] #<- get the destination (i.e. target node)
    w = e[2] #<- get the weight of the edge
# Add a source node with its information.
    network_ex.add_node(src, src, title = src)
# Add a destination node with its information.
    network_ex.add_node(dst, dst, title = dst)
# Add an edge between source and destination nodes with weight w.
    network_ex.add_edge(src, dst, value = w)


# In[ ]:


print(network_ex.nodes[0:5])
print(network_ex.edges[0:5])
print(network_ex.shape)


# #### Task 5
# ##### Get the neighbor map for each node.
# ##### Print the document IDs that are most similar to document 25.

# #### Result:

# In[ ]:


# Get a list of node neighbors.
neighbor_map = network_ex.get_adj_list()

# Show documents most similar to document 25.
print(neighbor_map[25])


# #### Task 6 
# ##### Add the neighbor node information into the hover over tooltip.
# ##### Print information of the node 5.
# ##### Save the network graph as `UN_similar_snippets` and show it in browser.

# #### Result:

# In[ ]:


# Add neighbor data to node hover data.
for node in network_ex.nodes:
    title = "Most similar articles: <br>"
    neighbors = list(neighbor_map[node["id"]])
    title = title + "<br>".join(str(neighbor) for neighbor in neighbors)
    node["title"] = title

print(network_ex.nodes[5])


# In[ ]:


# Save html and show graph in browser.
network_ex.show(plot_dir + "/UN_similar_snippets.html")


# #### Task 7
# ##### Hover over a node of your choice to see the list of all its neighbors. For example, node 924 is used below.
# ##### Print the articles from the edgelist `edgelist_df_ex` with their weights.
# ##### Look up the articles closest to the node and print them.
# ##### Modify the graph appearance by using `physics` parameter and re-save the graph.
# ##### Optional: Try using `nodes` and `edges` parameter to change the appearance of the graph.

# #### Result:

# In[ ]:


edgelist_df_subset_ex = edgelist_df_ex.query("source==924")
print(edgelist_df_subset_ex)


# In[ ]:


print(edgelist_df_subset_ex)


# In[ ]:


print(UN.iloc[924, 0])


# In[ ]:


print(UN.iloc[450, 0])


# In[ ]:


print(UN.iloc[914, 0])
# We can see that these 3 articles are similar, because their snippets all start and end the same way: Development Credit Agreement...


# In[ ]:


# Show buttons to modify the look.
network_ex.show_buttons(filter_=['physics'])


# In[ ]:


# Save html and show graph in browser.
network_ex.show(plot_dir+"/UN_similar_snippets.html")


# ## Exercise 3

# #### Task 1 
# ##### Compute the distance matrix `distance_ex` from `similarity_ex`.
# ##### Create the linkage matrix based on `distance_ex` and print the first 10 rows.
# ##### Print the shape of the matrix and the first 4 links.
# ##### Print the 110th link. Which clusters are linked? What is the distance between them? How many observations are there in the new cluster?
# 

# #### Result:

# In[ ]:


# Compute distance matrix by subtracting similarity from 1.
distance_ex = 1 - similarity_ex

# Define the `linkage_matrix` using `ward` clustering algorithm.
linkage_matrix_ex = ward(distance_ex)
print(linkage_matrix_ex[0:10])

# Print shape of the matrix.
print(linkage_matrix_ex.shape)
    
print(linkage_matrix_ex[0:4])

#Print the 110th link in the matrix.
print(linkage_matrix_ex[109])


# #### Task 2 
# ##### Visualize the hierarchical clusters with right orientation and leaf font size 14. Set figsize to (15, 40).
# 

# #### Result:

# In[ ]:


# Now we can plot the hierarchical clusters.
fig, axes = plt.subplots(figsize = (15, 40))
axes = dendrogram(linkage_matrix_ex,
                  orientation = "right",
                  labels = valid_snippets_ex,
                  leaf_font_size = 11)


# #### Task 3 
# ##### Split the dendrogram based on maximum clusters. Set the maximum number of clusters named `k` as 3.

# #### Result:

# In[ ]:


# Set k - the max number of clusters.
k = 3

# Get cluster labels for each snippet.
cluster_labels = fcluster(linkage_matrix_ex, #<- linkage matrix
                          k, #<- max number of clusters
                          criterion = 'maxclust') #<- criterion maxclust
print(cluster_labels)


# #### Task 4 
# ##### Create variable with valid snippets of `UN` and name as ` UN_valid_articles`.
# ##### Add `cluster_labels` to `UN_valid_articles` and name the column as `hclust_label`.
# ##### Sort `doc_topic_df_ex` by `doc_id` and save.
# ##### Add a column called `LDA_topic_label` to `UN_valid_articles` from `best_topic` in `doc_topic_df_ex`.
# ##### Save the plot and the data in png and csv format respectively.

# #### Result:

# In[ ]:


UN_valid_articles = UN.loc[valid_snippets_ex]
UN_valid_articles['hclust_label'] = cluster_labels
doc_topic_df_ex = doc_topic_df_ex.sort_values(by = "doc_id")
UN_valid_articles['LDA_topic_label'] = doc_topic_df_ex['best_topic']

fig.savefig(plot_dir + '/UN_hclust.png')
UN_valid_articles.to_csv(data_dir + '/UN_snippets_with_cluster_labels.csv')


# In[ ]:




