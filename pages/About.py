import streamlit as st
import pandas as pd

# Set file path
data_path = "data/"
materials_path = "materials/"

# Load the Excel file
topic_desc_bertopic_path = f"{materials_path}topic_descriptions_BERTopic.csv"
topic_desc_bertopic = pd.read_csv(topic_desc_bertopic_path)

topic_desc_lda_path = f"{materials_path}topic_descriptions_LDA.csv"
topic_desc_lda = pd.read_csv(topic_desc_lda_path)


# Iterate through each row in the DataFrame and display the description of a topic
def display_topic_descriptions(model_type="BERTopic", topic_id=0):
    # Check model type
    if model_type == "BERTopic":
        topic_desc_df = topic_desc_bertopic
    else:
        topic_desc_df = topic_desc_lda

    # Display the information based on the model type
    for index, row in topic_desc_df.iterrows():
        topic_number = row["ID"]
        topic_name = row["Topic"]
        representative_words = row["Representative Words"]
        description = row["Description"]

        # Display the topic information
        tab_id, tab_topic, tab_words = st.columns(
            [0.13, 0.37, 0.5], gap="small", vertical_alignment="top"
        )
        with tab_id:
            st.markdown(
                f"<h3>Topic {topic_number}</h3>",
                unsafe_allow_html=True,
            )
        with tab_topic:
            st.markdown(
                f"<h3>{topic_name}</h3>",
                unsafe_allow_html=True,
            )
        with tab_words:
            st.markdown(
                f"<h5 style='margin-top:10px'>{representative_words}</h5>",
                unsafe_allow_html=True,
            )
            # st.write(f"**Top Words: {representative_words}**")
        st.write(description)
        st.markdown("---")


# Header
st.title(f"Computer Science Research Topics")
tab1, tab2 = st.tabs(["BERTopic", "LDA"])

with tab1:
    display_topic_descriptions()

with tab2:
    display_topic_descriptions(model_type="LDA")
