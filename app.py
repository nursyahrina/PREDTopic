import streamlit as st
import pandas as pd
import spacy
from gensim import corpora
from gensim.models import LdaModel

from utils.preprocessing import single_text_preprocessing
from utils.visualization import (
    visualize_topic_over_time,
    visualize_top10words,
    visualize_topic_distribution,
    create_colored_text,
    print_topic_colors,
    display_wordcloud,
)

# Set theme
st.set_page_config(page_title="PREDTopic App", page_icon=":bar_chart:", layout="wide")

# Set file path
data_path = "data/"
lda_model_path = "models/lda/"

# Load prepared documents
data_df = pd.read_csv(f"{data_path}prepared_data.csv")

# Load LDA-clustered documents
topic_df_lda = pd.read_csv(f"{data_path}topic_probabilities_BestLDA-BoW.csv")

# Merge dataframes
topic_df_lda = topic_df_lda.join(
    data_df.drop(columns=["DOI", "Year", "Title", "Text"]), how="left"
)

# Load spaCy model for visualization
nlp = spacy.load("en_core_web_sm")

# Load LDA model
best_lda_model_path = f"{lda_model_path}best_lda_model"
lda_model = LdaModel.load(best_lda_model_path)
dictionary = corpora.Dictionary.load(f"{best_lda_model_path}.id2word")

# Best Topic
best_topic = topic_df_lda["Top Topic ID"].value_counts().idxmax()
best_topic_name = topic_df_lda["Top Topic Name"].value_counts().idxmax()
num_topics = lda_model.num_topics


# Innitialize state
def init_state():
    if f"Topic {best_topic}" not in st.session_state:
        st.session_state.selected_topic = f"Topic {best_topic}"
    if f"BERTopic" not in st.session_state:
        st.session_state.selected_model = f"BERTopic"


# Change selection state
def change_select_topic(new_topic):
    st.session_state.selected_topic = new_topic


# Inisialisasi state
init_state()

# Display header
container_header = st.empty()
with container_header.container():
    header_col1, header_col2, header_col3 = st.columns([0.6, 0.2, 0.2], gap="medium")
    with header_col1:
        st.title("Computer Science Research Topics")
    with header_col2:
        # Model selection
        container_select_model = st.empty()
        options = ["BERTopic", "LDA"]
        model_type = container_select_model.selectbox(
            label="Choose topic",
            options=options,
            index=options.index(st.session_state.selected_model),
        )
    with header_col3:
        # Topic selection
        options = [f"Topic {i}" for i in range(0, num_topics)]
        container_select_topic = st.empty()
        topic_id = container_select_topic.selectbox(
            label="Choose topic",
            options=options,
            index=options.index(st.session_state.selected_topic),
        )
        topic_id = int(topic_id.split(" ")[1])


# Function to display topic visualizations include representative words and documents
def display_topic_viz_docs(topic_id, container):
    if topic_id == best_topic:  # default display
        topic_df_to_show = topic_df_lda[topic_df_lda["Top Topic ID"] == best_topic]
        header = f"### :trophy: Top Topic: Topic {best_topic}\n{best_topic_name}"
    else:
        topic_df_to_show = topic_df_lda[topic_df_lda["Top Topic ID"] == topic_id]
        header = f'**[Topic {topic_id}]** {topic_df_to_show["Top Topic Name"].value_counts().idxmax()}'

    top_10_docs = (
        topic_df_to_show.sort_values("Top Topic Probability", ascending=False)
        .head(10)
        .reset_index()
    )

    with container.container():
        container_top = st.empty()
        with container_top.container():
            st.markdown(header)
        container_mid = st.container()
        # Create columns to part the display
        main_col1, main_col2 = st.columns([0.6, 0.4])
        with main_col1:
            st.plotly_chart(
                visualize_topic_over_time(topic_id, topic_df_to_show),
                use_container_width=True,
            )

        with main_col2:
            st.plotly_chart(
                visualize_top10words(model=lda_model, topic_id=topic_id),
                use_container_width=True,
            )
        # Display wordcloud
        st.pyplot(display_wordcloud(lda_model, topic_id))
        st.markdown("<br>", unsafe_allow_html=True)

        # Display representative documents
        st.markdown(f"##### Top 10 Representative Documents of Topic {topic_id}")
        for index, row in top_10_docs.iterrows():
            index_string = str(index + 1)
            expander = st.empty()
            with expander.expander(f'[{index_string}] {row["Title"]}'):
                data_col1, data_col2, data_col3 = st.columns([0.06, 0.8, 0.14])
                with data_col1:
                    st.markdown("<small>No.<small>", unsafe_allow_html=True)
                    st.write(index + 1)
                with data_col2:
                    st.markdown("<small>(Year) Title:<small>", unsafe_allow_html=True)
                    st.markdown(f'##### ({row["Year"]}) {row["Title"]}')
                with data_col3:
                    str_prob = f'{row["Top Topic Probability"]:.4f}'
                    st.markdown("<small>Probability:<small>", unsafe_allow_html=True)
                    st.write(float(str_prob))

                st.markdown(f'**DOI: [{row["DOI"]}](https://doi.org/{row["DOI"]})**')
                st.write("##### Abstract")
                st.write(row["Abstract"])
                if row["Author Keywords"] != "":
                    st.write("##### Author Keywords")
                    st.write(row["Author Keywords"])
    return container_top, container_mid


# Display main container
container_main = st.empty()
# Load main container
display_topic_viz_docs(topic_id, container_main)


# Function for new prediction
def predict_new_text_LDA(new_text, lda_model):
    # Preprocessed new text
    processed_new_text = single_text_preprocessing(new_text)

    # Convert new text to BoW (feature extraction)
    new_bow = dictionary.doc2bow(processed_new_text)

    # Get topic distribution of the new text (prediction)
    new_topic_distribution, per_word_topics, _ = lda_model.get_document_topics(
        new_bow, per_word_topics=True
    )

    # Sort topic distribution
    sorted_topic_distribution = sorted(
        new_topic_distribution, key=lambda x: x[1], reverse=True
    )
    topic_id = sorted_topic_distribution[0][0]

    # Create doc for visualization
    doc = nlp(new_text)

    return topic_id, sorted_topic_distribution, per_word_topics, doc


with st.sidebar:
    st.header("Topic Prediction for New Research Idea")
    # Text input for research description
    text = st.text_area("Describe your research idea here", height=250, max_chars=1500)
    # Button to choose model type
    st.write("Predict using:")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("BERTopic", use_container_width=True):
            st.write("Coming soon...")
            # st.switch_page(f"app.py/?model=BERTopic&topic={topic_pred}")
    with col2:
        if st.button("LDA", use_container_width=True):
            topic_id, new_topic_distribution, per_word_topics, doc = (
                predict_new_text_LDA(text, lda_model)
            )
            st.markdown(
                f"**Topic {topic_id}** is most likely with a probability of **{new_topic_distribution[0][1]:.4f}**!"
            )
            change_select_topic(f"Topic {topic_id}")

            container_top, container_mid = display_topic_viz_docs(
                topic_id, container_main
            )

            with container_top.container():
                st.markdown(
                    f"<h3 style='text-align: center; margin-bottom: 24px'>Topic Distribution in Document (Prediction=Topic{topic_id})</h3>",
                    unsafe_allow_html=True,
                )

            with container_mid.container():
                st.markdown(
                    f"<p>{create_colored_text(doc, dictionary, per_word_topics)}</p>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<p>{print_topic_colors(num_topics)}</p>", unsafe_allow_html=True
                )
                st.plotly_chart(visualize_topic_distribution(new_topic_distribution))
