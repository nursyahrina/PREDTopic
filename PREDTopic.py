import streamlit as st
import pandas as pd
import spacy
import nltk
from gensim import corpora
from gensim.models import LdaModel
from bertopic import BERTopic

from utils.preprocessing import single_text_preprocessing
from utils.visualization import (
    visualize_topic_over_time,
    visualize_top10words_bertopic,
    visualize_top10words_lda,
    visualize_topic_distribution,
    visualize_wordcloud_bertopic,
    visualize_wordcloud_lda,
    create_colored_text,
    print_topic_colors,
)

# SETTINGS

# Set theme
st.set_page_config(page_title="PREDTopic App", page_icon=":bar_chart:", layout="wide")

# Set file path
data_path = "data/"
lda_model_path = "models/lda_model/"
bertopic_model_path = "models/bertopic_model"
materials_path = "materials/documentation/"


# LOAD DATA

# Load prepared documents
data_df = pd.read_csv(f"{data_path}prepared_data.csv")

# Load LDA-clustered documents and merge with dataset
topic_df_lda = pd.read_csv(f"{data_path}topic_probabilities_LDA-BoW.csv")
topic_df_lda = topic_df_lda.join(
    data_df.drop(columns=["DOI", "Year", "Title", "Text"]), how="left"
)

# Load BERTopic-clustered documents and merge with dataset
topic_df_bertopic = pd.read_csv(f"{data_path}topic_documents_BERTopic.csv")
topic_df_bertopic = topic_df_bertopic.join(
    data_df.drop(columns=["DOI", "Year", "Title", "Text"]), how="left"
)
# fill probability value same for all data in BERTopic model,
# since this model does not provide probability for each document.
topic_df_bertopic["Top Topic Probability"] = 1

# LOAD MODEL

# Load LDA model
best_lda_model_path = f"{lda_model_path}best_lda_model"
lda_model = LdaModel.load(best_lda_model_path)
dictionary = corpora.Dictionary.load(f"{best_lda_model_path}.id2word")

# Load BERTopic model
bertopic_model = BERTopic.load(f"{bertopic_model_path}")

# Ensure necessary NLTK data is downloaded
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")

# Load spaCy model for visualization
nlp = spacy.load("en_core_web_sm")


# LOAD ADDITIONAL MATERIALS (e.g., topic descriptions)
topic_desc_bertopic_path = f"{materials_path}topic_descriptions_BERTopic.csv"
topic_desc_bertopic = pd.read_csv(topic_desc_bertopic_path)

topic_desc_lda_path = f"{materials_path}topic_descriptions_LDA.csv"
topic_desc_lda = pd.read_csv(topic_desc_lda_path)


# GET BEST TOPIC

# Best Topic of LDA
best_topic_lda = topic_df_lda["Top Topic ID"].value_counts().idxmax()
best_topic_name_lda = topic_df_lda["Top Topic Name"].value_counts().idxmax()
num_topics_lda = 11

# Best Topic of BERTopic
best_topic_bertopic = topic_df_bertopic["Top Topic ID"].value_counts().idxmax()
best_topic_name_bertopic = topic_df_bertopic["Top Topic Name"].value_counts().idxmax()
num_topics_bertopic = 13


# STATE HANDLING
# Default
model_type = "BERTopic"
best_topic = best_topic_bertopic
topic_id = best_topic
best_topic_name = best_topic_name_bertopic
num_topics = num_topics_bertopic
topic_df = topic_df_bertopic


# Innitialize state
def init_state():
    st.session_state.selected_topic = f"Topic {best_topic_bertopic}"
    st.session_state.selected_model = "BERTopic"


# START APP


# Function to display header
def display_header(container):
    with container.container():

        container_head = st.empty()
        with container_head:
            header_title, select_model, select_topic = st.columns(
                [0.6, 0.2, 0.2], gap="medium"
            )
            with header_title:
                st.title(f"Computer Science Research Topics")
            with select_model:
                # Model selection
                container_select_model = st.empty()
                options = ["BERTopic", "LDA"]
                model_selectbox = container_select_model.selectbox(
                    label="Choose topic",
                    options=options,
                    index=options.index(st.session_state.selected_model),
                    key="init_model",
                )

                # Update session state when a new option is slected
                st.session_state.selected_model = model_selectbox

                # Update topic selection when a new model type is selected
                if st.session_state.selected_model == "BERTopic":
                    st.session_state.selected_topic = f"Topic {best_topic_bertopic}"
                else:
                    st.session_state.selected_topic = f"Topic {best_topic_lda}"
            with select_topic:
                # Topic selection, set options according to the selected model type
                options = []
                if st.session_state.selected_model == "BERTopic":
                    options = [f"Topic {i}" for i in range(0, num_topics_bertopic)]
                else:
                    options = [f"Topic {i}" for i in range(0, num_topics_lda)]
                container_select_topic = st.empty()
                topic_selectbox = container_select_topic.selectbox(
                    label="Choose topic",
                    options=options,
                    index=options.index(st.session_state.selected_topic),
                    key="init_topic",
                )
                # Update session state when a new option is slected
                st.session_state.selected_topic = topic_selectbox

                # Update topic id when a new topic is selected
                topic_id = int(topic_selectbox.split(" ")[1])

        return topic_id


# Function to display topic visualizations include representative words and documents
def display_topic_viz_docs(topic_id, container):
    # Set data by model type selected
    if st.session_state.selected_model == "BERTopic":
        best_topic = best_topic_bertopic
        best_topic_name = best_topic_name_bertopic
        topic_df = topic_df_bertopic
    else:
        best_topic = best_topic_lda
        best_topic_name = best_topic_name_lda
        topic_df = topic_df_lda

    # Get topic informations
    if topic_id == best_topic:  # default display
        topic_df_to_show = topic_df[topic_df["Top Topic ID"] == best_topic]
        header = f"### :trophy: Top Topic: Topic {best_topic}\n{best_topic_name}"
    else:
        topic_df_to_show = topic_df[topic_df["Top Topic ID"] == topic_id]
        header = f'**[Topic {topic_id}]** {topic_df_to_show["Top Topic Name"].value_counts().idxmax()}'

    # Get 10 representative documents
    top_10_docs = topic_df_to_show.sample(10, random_state=42)
    top_10_docs.reset_index(drop=True, inplace=True)

    # Get 10 most probable documents for LDA model type
    if st.session_state.selected_model == "LDA":
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
            if st.session_state.selected_model == "BERTopic":
                st.plotly_chart(
                    visualize_top10words_bertopic(
                        model=bertopic_model,
                        topic_id=topic_id,
                    ),
                    use_container_width=True,
                )
            else:
                st.plotly_chart(
                    visualize_top10words_lda(
                        model=lda_model,
                        topic_id=topic_id,
                    ),
                    use_container_width=True,
                )

        # Display wordcloud
        if st.session_state.selected_model == "BERTopic":
            st.pyplot(visualize_wordcloud_bertopic(bertopic_model, topic_id))
        else:
            st.pyplot(visualize_wordcloud_lda(lda_model, topic_id))
        st.markdown("<br>", unsafe_allow_html=True)

        st.divider()

        # Display the description of the topic
        if st.session_state.selected_model == "BERTopic":
            st.markdown(
                f"<h5 style:'margin-top:12px'>Topic {topic_id}: {topic_desc_bertopic['Topic'][topic_id]}</h5>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<p style:'text-align: justify; text-justify: inter-word;'>{topic_desc_bertopic['Description'][topic_id]}</p>",
                unsafe_allow_html=True,
                # f"**Description**: {topic_desc_bertopic['Description'][topic_id]}"
            )

        else:
            st.markdown(f"##### Topic {topic_id}: {topic_desc_lda['Topic'][topic_id]}")
            st.markdown(f"**Description**: {topic_desc_lda['Description'][topic_id]}")

        st.divider()

        # Display representative documents

        st.markdown(
            f"<h5 style:'margin-top:12px;'>Representative Documents of Topic {topic_id}</h5>",
            unsafe_allow_html=True,
        )
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
                    if st.session_state.selected_model == "LDA":
                        str_prob = f'{row["Top Topic Probability"]:.4f}'
                        st.markdown(
                            "<small>Probability:<small>", unsafe_allow_html=True
                        )
                        st.write(float(str_prob))
                    else:
                        pass

                st.markdown(f'**DOI: [{row["DOI"]}](https://doi.org/{row["DOI"]})**')
                st.write("##### Abstract")
                st.write(row["Abstract"])
    return container_top, container_mid


# Inisialisasi state
init_state()

# Initialize header
container_header = st.empty()
# Initialize main container
container_main = st.empty()

# Load header
topic_id = display_header(container_header)
# Load main container
display_topic_viz_docs(topic_id, container_main)


# Function for new prediction using BERTopic
def predict_topic_bertopic(new_text, bertopic_model):
    [topic_id], [probability] = bertopic_model.transform([new_text])
    return topic_id, probability


# Function for new prediction using LDA
def predict_topic_lda(new_text, lda_model):
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
    st.markdown(f"### Topic Prediction for New Research Idea", unsafe_allow_html=True)
    # Text input for research description
    text = st.text_area("Describe your research idea here", height=250, max_chars=3000)
    # Button to choose model type
    st.write("Predict using:")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("BERTopic", use_container_width=True):
            topic_id, probability = predict_topic_bertopic(text, bertopic_model)

            # Update selectbox and state
            st.session_state.selected_model = "BERTopic"
            st.session_state.selected_topic = f"Topic {topic_id}"

            # Remove header
            container_header.empty()

            # Display prediction results and reload main container
            container_top, container_mid = display_topic_viz_docs(
                topic_id, container_main
            )

            with container_top.container():
                st.markdown(
                    f"<h3 style='text-align: center; margin-bottom: 12px'>Prediction by BERTopic Model = Topic {topic_id}</h3>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<h5 style='text-align: center; margin-bottom: 24px'>The text is most likely related to Topic {topic_id} with a {probability*100:.2f}% probability</h5>",
                    unsafe_allow_html=True,
                )

    with col2:
        if st.button("LDA", use_container_width=True):
            topic_id, new_topic_distribution, per_word_topics, doc = predict_topic_lda(
                text, lda_model
            )

            # Update selectbox and state
            st.session_state.selected_model = "LDA"
            st.session_state.selected_topic = f"Topic {topic_id}"

            # Remove header
            container_header.empty()

            # Display prediction results and reload main container
            container_top, container_mid = display_topic_viz_docs(
                topic_id, container_main
            )

            with container_top.container():
                st.markdown(
                    f"<h3 style='text-align: center; margin-bottom: 12px'>Prediction by LDA Model = Topic {topic_id}</h3>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<h5 style='text-align: center; margin-bottom: 8px'>The text is most likely related to Topic {topic_id} with a  {new_topic_distribution[0][1]*100:.2f}% probability</h5>",
                    unsafe_allow_html=True,
                )

            # Display detail results of prediction word by word using topic distributions
            with container_mid.container():
                expander = st.empty()
                with expander.expander("Topic Distribution", expanded=True):
                    st.markdown(
                        f"<p>{create_colored_text(doc, dictionary, per_word_topics)}</p>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<p>{print_topic_colors(num_topics_lda)}</p>",
                        unsafe_allow_html=True,
                    )
                    st.plotly_chart(
                        visualize_topic_distribution(new_topic_distribution)
                    )
