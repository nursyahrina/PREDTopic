import streamlit as st

st.set_page_config(
    page_title="About | PREDTopic",
    page_icon=":bar_chart:",
    layout="wide",
)

# Set file path
data_path = "data/"
materials_path = "materials/documentation/"
images_path = "materials/images/"

# Load the Excel file
app_layout_file = f"{materials_path}app_layout.md"
with open(app_layout_file, "r") as f:
    app_layout = f.read()

feature_list_file1 = f"{materials_path}feature_list_1.md"
with open(feature_list_file1, "r") as f:
    feature_list1 = f.read()

feature_list_file2 = f"{materials_path}feature_list_2.md"
with open(feature_list_file2, "r") as f:
    feature_list2 = f.read()

about_models_file = f"{materials_path}about_models.md"
with open(about_models_file, "r") as f:
    about_models = f.read()

trend_analysis_LDA_file = f"{materials_path}trend_analysis_LDA.md"
with open(trend_analysis_LDA_file, "r") as f:
    trend_analysis_LDA = f.read()

trend_analysis_BERTopic_file = f"{materials_path}trend_analysis_BERTopic.md"
with open(trend_analysis_BERTopic_file, "r") as f:
    trend_analysis_BERTopic = f.read()

# Header
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["About Models", "Trend Analysis", "App Features", "User Guide", "About Author"]
)

with tab1:
    st.markdown(about_models, unsafe_allow_html=True)

with tab2:
    st.markdown(
        """### Research Trends in Computer Science (2019-2023)

#### Methodology

The trend analysis was conducted using LDA and BERTopic models applied to 4,892 computer science research article metadata entries from Emerald Insight (2019-2023). We analyzed trends by calculating the number of publications per topic for each year and computing year-over-year growth rates using the formula: Growth Rate = (Current Year Count - Previous Year Count) / Previous Year Count. 

Average annual growth rates were determined for topics showing consistent trends. Significant changes were identified by noting topics with annual fluctuations of 20% or more. We also assessed each topic's overall importance by comparing its total publication count across all years. 

This quantitative analysis was then interpreted within the context of broader trends in computer science and relevant global events, providing insights into the evolving landscape of computer science research over the five-year period.""",
        unsafe_allow_html=True,
    )
    st.markdown("---", unsafe_allow_html=True)
    st.markdown("#### BERTopic Model Insights", unsafe_allow_html=True)
    st.image(
        f"{images_path}heatmap_BERTopic.png",
        caption="Fig 2.1 Topic Trend Evolution Heatmap using BERTopic Model",
        use_column_width=True,
        output_format="PNG",
    )
    st.markdown(trend_analysis_BERTopic, unsafe_allow_html=True)
    st.markdown("---", unsafe_allow_html=True)
    st.markdown("#### LDA Model Insights", unsafe_allow_html=True)
    st.image(
        f"{images_path}heatmap_LDA.png",
        caption="Fig 2.2 Topic Trend Evolution Heatmap using LDA Model",
        use_column_width=True,
        output_format="PNG",
    )
    st.markdown(trend_analysis_LDA, unsafe_allow_html=True)
    st.markdown(
        """--- 
        
Overall, these trends demonstrate a shift from traditional topics like project management towards a greater focus on business analytics, blockchain, IoT, and predictive techniques. This evolution reflects technological advancements and changing industry needs, highlighting the dynamic nature of computer science research in responding to real-world challenges and opportunities.

The analysis reveals not only the dominant research areas but also emerging fields that are gaining traction. It's important to note that while some topics show consistent growth or decline, others exhibit more volatile patterns, which may be indicative of rapidly evolving research interests or responses to external factors such as technological breakthroughs or global events like the COVID-19 pandemic.""",
        unsafe_allow_html=True,
    )

with tab3:
    st.markdown("### What's in PREDTopic?", unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.markdown(app_layout, unsafe_allow_html=True)
    with col2:
        st.image(
            f"{images_path}app_features.png",
            caption="Fig 3.1 PREDTopic Features",
            use_column_width=True,
            output_format="PNG",
        )
    st.markdown("#### Here are the key features:", unsafe_allow_html=True)
    col3, col4 = st.columns(2, gap="medium")
    with col3:
        st.markdown(feature_list1, unsafe_allow_html=True)
    with col4:
        st.markdown(feature_list2, unsafe_allow_html=True)


with tab4:
    st.markdown("### Step-by-Step Usage Instructions", unsafe_allow_html=True)
    st.markdown(
        """#### 1. **Starting the Application**

When you first launch PREDTopic, it will automatically load the data and display the **Top Topic** along with related visualizations in the **Main Section**.""",
        unsafe_allow_html=True,
    )
    st.image(
        f"{images_path}default_view.png",
        caption="Fig 4.1 Default View of The Main Page",
        use_column_width=True,
        output_format="PNG",
    )
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.markdown(
            """#### 2. **Selecting a Topic Model**
            
   If you wish to explore topics identified by a different model, go to the top right corner and use the **Select Topic Model** dropdown. Choose either "LDA" or "BERTopic" depending on your preference. The application will refresh the visualizations based on the selected model.""",
            unsafe_allow_html=True,
        )
        st.image(
            f"{images_path}model_options.png",
            caption="Fig 4.2 Choosing a Topic Model",
            use_column_width=True,
            output_format="PNG",
        )
    with col2:
        st.markdown(
            """#### 3. **Selecting a Topic**
   
   Once a model is selected, you can choose a specific topic to explore. Use the **Select Topic** dropdown to pick from the available topics. The visualizations in the **Main Section** will update to reflect the data for the selected topic.""",
            unsafe_allow_html=True,
        )
        st.image(
            f"{images_path}topic_options.png",
            caption="Fig 4.3 Choosing a Topic Based on The Selected Model",
            use_column_width=True,
            output_format="PNG",
        )
    st.markdown("#### 4. **Predicting a Research Topic**", unsafe_allow_html=True)
    col3, col4 = st.columns(2, gap="medium")
    with col3:
        st.markdown(
            """To get a topic prediction based on your research idea, enter your text into the **Predict Research Idea** input box located in the **Sidebar**. 
   
   Click either the "LDA" or "BERTopic" button depending on which model you want to use for the prediction. The application will analyze your text and display the most relevant topic in the **Main Section**.""",
            unsafe_allow_html=True,
        )
        st.image(
            f"{images_path}input_text.png",
            caption="Fig 4.4 Input Text Describing Research Idea and Choose Model for Prediction",
            use_column_width=True,
            output_format="PNG",
        )
    with col4:
        st.markdown(
            """The prediction result will be shown at the top of the **Main Section**, declaring which topic is most relevant to your research idea and it's probability.

   The **Main Section** will update to reflect the predicted topic and its associated visualizations.""",
            unsafe_allow_html=True,
        )
        st.image(
            f"{images_path}prediction_BERTopic.png",
            caption="Fig 4.5 Prediction Result by BERTopic Model",
            use_column_width=True,
            output_format="PNG",
        )
        st.markdown(
            """Specifically for results from LDA, the topic distribution in the input text will also be displayed.""",
            unsafe_allow_html=True,
        )
        st.image(
            f"{images_path}prediction_LDA.png",
            caption="Fig 4.6 Prediction Result by LDA Model",
            use_column_width=True,
            output_format="PNG",
        )

with tab5:
    st.markdown("### About the Author", unsafe_allow_html=True)
    st.write(
        """
"""
    )
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.markdown("#### Author", unsafe_allow_html=True)
        st.write("Nursyahrina")

    with col2:
        st.markdown("#### Research Advisor 1", unsafe_allow_html=True)
        st.write("Prof. Dr. Sarjon Defit, S.Kom., M.Sc")

    with col3:
        st.markdown("#### Research Advisor 2", unsafe_allow_html=True)
        st.write("Dr. Rini Sovia, S.Kom., M.Kom")

    st.write(
        """
"""
    )
    st.write(
        "This application is part of the output from my thesis research titled: ***Analisis Tren Penelitian Bidang Ilmu Komputer dengan Metode BERTopic dan LDA*** (Research Trends Analysis in Computer Science Using BERTopic and LDA)"
    )

    st.markdown("---")
    st.write(":mortar_board: **Faculty**: Fakultas Ilmu Komputer")
    st.write(
        ':school: **University**: Universitas Putra Indonesia "YPTK", Padang, 25221, Indonesia'
    )
    st.write(":email: **Email**: nursyahrina17@gmail.com")
