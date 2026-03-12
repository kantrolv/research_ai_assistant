from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import umap
import matplotlib.pyplot as plt

# Load the SentenceTransformer model
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
except:
    model = None

# Mock database of research papers to compare against
paper_database = [
    {
        "title": "Attention Is All You Need", 
        "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely."
    },
    {
        "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "abstract": "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers."
    },
    {
        "title": "Graph Representation Learning: A Survey",
        "abstract": "Machine learning on graphs is an important and ubiquitous task with applications ranging from drug design to friendship recommendation in social networks. The primary challenge in this domain is finding a way to represent, or encode, graph structure so that it can be easily exploited by machine learning models."
    },
    {
        "title": "A Survey on Topic Modeling in Text Mining",
        "abstract": "Topic modeling is a machine learning technique that automatically analyzes text data to determine cluster words for a set of documents. This is known as unsupervised machine learning because it doesn't require a predefined list of tags or training data that's been previously classified by humans."
    },
    {
        "title": "Generative Adversarial Nets",
        "abstract": "We propose a new framework for estimating generative models via an adversarial process, in which we simultaneously train two models: a generative model G that captures the data distribution, and a discriminative model D that estimates the probability that a sample came from the training data rather than G."
    },
    {
        "title": "Dimensionality Reduction: A Comparative Review",
        "abstract": "High-dimensional data are ubiquitous in machine learning and data mining. Dimensionality reduction techniques, such as PCA, t-SNE, and UMAP, are widely used to map high-dimensional data to a lower-dimensional space while preserving its inherent structure."
    },
    {
        "title": "Deep Residual Learning for Image Recognition",
        "abstract": "Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions."
    },
    {
        "title": "Adam: A Method for Stochastic Optimization",
        "abstract": "We introduce Adam, an algorithm for first-order gradient-based optimization of stochastic objective functions, based on adaptive estimates of lower-order moments. The method is straightforward to implement, is computationally efficient, has little memory requirements, is invariant to diagonal rescaling of the gradients, and is well suited for problems that are large in terms of data and/or parameters."
    }
]

# Pre-compute embeddings for the mock database
if model:
    db_embeddings = model.encode([p["abstract"] for p in paper_database])
else:
    db_embeddings = np.array([])

def get_similar_papers(query_text, top_n=5):
    if not model:
        return []
    
    query_emb = model.encode([query_text])
    similarities = cosine_similarity(query_emb, db_embeddings)[0]
    
    # Sort by similarity in descending order
    top_indices = np.argsort(similarities)[::-1][:top_n]
    
    results = []
    for idx in top_indices:
        results.append({
            "title": paper_database[idx]["title"],
            "abstract": paper_database[idx]["abstract"],
            "similarity": similarities[idx]
        })
    return results

def get_query_embedding(query_text):
    if not model:
        return np.zeros((1, 384))
    return model.encode([query_text])

def generate_topic_map(query_embedding):
    """
    Generates a 2D UMAP scatter plot of the predefined paper database along with the uploaded query paper.
    Colors by a dynamic KMeans clustering over the sentence embeddings to simulate 'Color points by cluster'.
    """
    if db_embeddings.size == 0 or query_embedding.size == 0:
        fig, ax = plt.subplots()
        ax.axis('off')
        return fig

    # Combine database + query
    all_emb = np.vstack([db_embeddings, query_embedding])
    
    # Optional: Perform KMeans on the DB+Query to assign colors
    from sklearn.cluster import KMeans
    kmeans_sim = KMeans(n_clusters=3, random_state=42)
    labels = kmeans_sim.fit_predict(all_emb)
    
    # UMAP to reduce into 2D
    n_neighbors_val = min(10, max(2, all_emb.shape[0] - 1))
    reducer = umap.UMAP(
        n_neighbors=n_neighbors_val,
        min_dist=0.1,
        n_components=2,
        random_state=42
    )
    embedding_2d = reducer.fit_transform(all_emb)
    
    fig, ax = plt.subplots(figsize=(9, 6))
    
    # Plot Database points
    scatter = ax.scatter(
        embedding_2d[:-1, 0], 
        embedding_2d[:-1, 1], 
        c=labels[:-1], 
        cmap='viridis', 
        s=150, 
        alpha=0.7, 
        edgecolors='w',
        label='Database Papers'
    )
    
    # Plot Query point
    ax.scatter(
        embedding_2d[-1:, 0], 
        embedding_2d[-1:, 1], 
        color='red', 
        s=250, 
        marker='*', 
        edgecolor='black',
        label='Your Uploaded Paper'
    )
    
    ax.legend()
    fig.patch.set_alpha(0.0)
    ax.axis('off')
    
    return fig
