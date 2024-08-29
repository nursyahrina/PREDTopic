import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud


# Plot the topic over time
def visualize_topic_over_time(topic_id, topic_df_to_show):
    # Group by Year and Top Topic, then count documents
    topic_year_counts = (
        topic_df_to_show.groupby(["Year", "Top Topic Name"])
        .size()
        .reset_index(name="Document_Count")
    )
    # Plotting using plotly line plot
    fig = px.line(
        topic_year_counts,
        x="Year",
        y="Document_Count",
        color="Top Topic Name",
        title=f"Topic {topic_id} Over The Years",
        labels={
            "Year": "Year",
            "Document_Count": "Number of Documents",
            "Top Topic Name": "Topic ID & Top Words",
        },
        height=400,
    )

    fig.update_traces(mode="lines")
    fig.update_layout(
        xaxis=dict(type="category"),
        showlegend=False,
        template="plotly_white",
    )
    return fig


def visualize_top10words_lda(model, topic_id):
    # Build dataframe of word_prob for topic_id
    word_prob = model.show_topics(
        num_topics=23, num_words=10, log=False, formatted=False
    )[topic_id][1]
    df = pd.DataFrame(word_prob, columns=["Word", "Probability"]).sort_values(
        "Probability"
    )

    # Create bar chart using plotly
    fig = px.bar(
        df,
        x="Probability",
        y="Word",
        title=f"Top 10 Words of Topic {topic_id}",
        template="plotly_white",
        orientation="h",
        color="Probability",
        height=400,
        color_continuous_scale="Viridis",
    )
    # Update layout to remove color bar
    fig.update_layout(
        coloraxis_showscale=False,
    )
    return fig


def visualize_top10words_bertopic(model, topic_id):
    # Get the topic's words and their c-TF-IDF scores
    words_scores = model.get_topic(topic_id)[:10]

    # Build dataframe from the words and scores
    df = pd.DataFrame(words_scores, columns=["Word", "c-TF-IDF Score"]).sort_values(
        "c-TF-IDF Score"
    )

    # Create bar chart using plotly
    fig = px.bar(
        df,
        x="c-TF-IDF Score",
        y="Word",
        title=f"Top 10 Words of Topic {topic_id}",
        template="plotly_white",
        orientation="h",
        color="c-TF-IDF Score",
        height=400,
        color_continuous_scale="Viridis",
    )

    # Update layout to remove color bar
    fig.update_layout(
        coloraxis_showscale=False,
    )

    # Display chart
    return fig


# Function to create wordcloud for a given topic
def visualize_wordcloud_lda(model, topic_id, num_words=50):
    # Get the top words for the topic
    topic_words = model.show_topic(topic_id, topn=num_words)
    wordcloud_dict = {word: freq for word, freq in topic_words}

    # Generate wordcloud
    wordcloud = WordCloud(
        width=1400, height=200, background_color="white"
    ).generate_from_frequencies(wordcloud_dict)

    plt.title(
        f"Top 50 Words of Topic {topic_id} in a Wordcloud",
        fontdict={"fontsize": 6, "fontweight": "semibold", "fontfamily": "sans-serif"},
    )
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")

    return plt


# Function to create wordcloud for a given topic in BERTopic
def visualize_wordcloud_bertopic(model, topic_id, num_words=50):
    # Get the top words for the topic and their c-TF-IDF scores
    topic_words_scores = model.get_topic(topic_id)

    # Create a dictionary of word frequencies
    wordcloud_dict = {word: score for word, score in topic_words_scores[:num_words]}

    # Generate wordcloud
    wordcloud = WordCloud(
        width=1400, height=200, background_color="white"
    ).generate_from_frequencies(wordcloud_dict)

    # Display the wordcloud
    plt.title(
        f"Top 50 Words of Topic {topic_id} in a Wordcloud",
        fontdict={"fontsize": 6, "fontweight": "semibold", "fontfamily": "sans-serif"},
    )
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")

    return plt


# Fungsi untuk memberi warna berdasarkan topik
# Fungsi untuk memberi warna berdasarkan topik
colors = [
    "#FFB3BA",  # Pastel Pink
    "#FFDFBA",  # Pastel Orange
    "#FFFFBA",  # Pastel Yellow
    "#BAFFC9",  # Pastel Green
    "#BAE1FF",  # Pastel Blue
    "#E3BFFF",  # Pastel Purple
    "#FFD1DC",  # Pastel Light Pink
    "#BFFCC6",  # Pastel Mint
    "#C9C9FF",  # Pastel Lavender
    "#FFCCF9",  # Pastel Magenta
    "#FDFD96",  # Pastel Lemon
]


def colorize(word, topic):
    return f'<span style="background-color: {colors[topic]};">{word}</span>'


def print_topic_colors(n_topics):
    text = ""
    for i in range(n_topics):
        text += f"{colorize(f'Topic {i}', i)} "
    return text


def visualize_topic_distribution(distribution):

    # Create a horizontal bar chart
    fig = go.Figure()

    for i, (topic_id, prob) in enumerate(distribution):
        fig.add_trace(
            go.Bar(
                y=["Document Topics"],  # A single bar divided into segments
                x=[prob],
                name=f"<b>Topic {topic_id}</b>",
                orientation="h",
                marker=dict(color=colors[topic_id]),
                hoverinfo="none",  # This disables default hoverinfo
                hovertemplate="<b>Topic ID:</b> %{text}<br><b>Probability:</b> %{x:.2%}",
                text=[f"{topic_id}"],  # Custom text to be used in hovertemplate
            )
        )

    # Update the layout
    fig.update_layout(
        barmode="stack",
        showlegend=False,
        xaxis_title="Probability",
        yaxis_title="",
        template="plotly_white",
        height=160,
        margin=dict(t=20),
    )

    return fig


def create_colored_text(doc, dictionary, per_word_topics):
    # Display colored text
    colored_text = ""
    for i, token in enumerate(doc):
        j = 0
        while (
            j < len(per_word_topics)
            and dictionary[per_word_topics[j][0]] != token.text.lower()
        ):
            j += 1
        if j < len(per_word_topics):  # found the token in word topic distribution
            word_topics = per_word_topics[j][1]
            if word_topics:
                colored_text += colorize(token.text, word_topics[0]) + " "
            else:
                colored_text += token.text + " "
        else:
            colored_text += token.text + " "

    return colored_text
