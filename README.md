Analyzed 200 Million Posts from publicly available Exorde dataset for identifying sentiments, key themes and primary emotions. 

Link to results: https://medium.com/@ritu.bansalrb00/the-emotional-landscape-of-the-internet-what-i-learnt-from-analyzing-200-million-social-media-4761e29b7104

File Information:
1. download_dataset.py - Creates a local copy of the dataset using Hugging Face package. Wouldn't recommend this since it takes up huge diskspace
2. preprocessing_script.py - Performs pre-processing on the local version of the dataset
3. grouping_script.py - Aggregates the dataset to prepare it for analysis
4. filter_politics.py - Streaming data to filter posts about politics to prepare for Named Entity Recognition task
5. named_entity_recognition.py - Used rule-based identification of named entities for sentiment/entropy analysis
6. eda.ipynb - Notebook for creating visualizations
