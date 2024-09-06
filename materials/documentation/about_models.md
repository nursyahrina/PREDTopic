### Topic Modeling: LDA and BERTopic

This application employs two advanced topic modeling techniques: Latent Dirichlet Allocation (LDA) and BERTopic. Both models were trained on a comprehensive dataset of computer science research articles metadata.


#### Dataset

- Source: Emerald Insight platform
- Number of entries: 4,892 metadata records
- Time span: 5 years (2019-2023)
- Key attributes: DOI, year, title, and abstract

---

#### Data Preparation and Preprocessing

For both models, the initial dataset underwent a thorough preparation process, including language filtering (English only), attribute selection, duplicate removal, and data completeness checks. The title and abstract fields were merged to create a unified text field for analysis.


##### LDA-specific Preprocessing
The LDA model required additional text preprocessing steps:
1. Case folding
2. Cleaning
3. Tokenization
4. Stop words removal
5. Lemmatization

These steps were followed by feature extraction using the Bag-of-Words (BoW) technique.


##### BERTopic Preprocessing
One of BERTopic's advantages is its ability to process raw text data. Therefore, it did not require the extensive preprocessing steps applied to the LDA input.

---

#### Model Evaluation and Selection

Both LDA and BERTopic models underwent rigorous evaluation to determine the optimal configuration for each method.


##### LDA Evaluation
- Experimented with different numbers of topics (≤ 20)
- Tested both BoW and TF-IDF feature extraction methods
- Selected based on coherence score and topic distribution quality


##### BERTopic Evaluation
- Tested various combinations of dimensionality reduction (UMAP, PCA, TruncatedSVD) and clustering (HDBSCAN, KMeans) techniques
- Experimented with different cluster sizes (30 ≤ min_cluster_size ≤ 100 for HDBSCAN, num_topics ≤ 20 for KMeans)
- Evaluated based on coherence score and topic distribution


The final selected models were:
1. LDA: LDA-BoW with 11 topics
2. BERTopic: TruncatedSVD-KMeans with 13 topics

---

#### Model Performance

##### Coherence Scores
- LDA (BoW) model: 0.42
- BERTopic (TruncatedSVD-KMeans) model: 0.49

The higher coherence score of BERTopic indicates its superior ability to generate more coherent and meaningful topics compared to LDA. This can be attributed to BERTopic's utilization of rich semantic representations from transformer-based language models, allowing for better capture of nuances and context in research texts.


##### Prediction Accuracy
- LDA model: 100%
- BERTopic model: 82.17%


It's crucial to note that these prediction accuracy scores were obtained by testing on the same dataset used for model training. This approach was necessary because:

1. Topic modeling is an unsupervised learning task, meaning the original data doesn't have predefined topic labels.
2. New, unseen data cannot be used for accuracy testing as there are no ground truth labels to compare against.

While LDA achieved perfect prediction accuracy on the training data, BERTopic's slightly lower accuracy is due to the use of an incomplete model implementation, necessitated by hardware limitations. A full BERTopic model implementation would likely achieve comparable accuracy.

However, it's important to emphasize that in the context of topic modeling, the concept of prediction accuracy is less relevant and potentially misleading. The primary goal of topic modeling is to uncover latent themes within a corpus, not to classify documents into predefined categories. Therefore, coherence scores and qualitative assessment of topic interpretability are generally more informative measures of model performance in real-world applications.

---

#### Topic Distribution

The LDA model (BoW) generated 11 topics, while the BERTopic model (TruncatedSVD-KMeans) produced 13 topics. BERTopic demonstrated a more balanced distribution of documents across topics, avoiding the formation of very small topics. This suggests BERTopic's robustness in identifying meaningful themes while minimizing noise or overly specific categorizations.

In conclusion, both models offer valuable insights into research trends in computer science. While LDA excels at identifying dominant themes, BERTopic provides a more nuanced view of topic variations within the dataset. The choice between these models depends on the specific analytical goals and the desired granularity of topic representation.