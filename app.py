import pickle
import pandas as pd
import requests
import streamlit as st
# from nltk.stem.porter import PorterStemmer

# import dataset stored in pickle file
a = pickle.load(open('anime.pkl', 'rb'))
ndf = pd.read_pickle('anime.pkl')

st.image('https://i.pinimg.com/736x/65/c1/66/65c16665f39cbb90161dcf140c6cfce7.jpg')


def fetch_poster(movie_id):
    url = "https://api.jikan.moe/v3/anime/{}".format(movie_id)
    data = requests.get(url)
    data = data.json()

    poster_path = data['image_url']
    return poster_path


def recommend(anime):
    anime_index = ndf[ndf['Name'] == anime].index[0]
    distances = sorted(list(enumerate(similarity[anime_index])), reverse=True, key=lambda x: x[1])[1:6]
    # recommendations
    recommendations = []
    recommend_poster = []

    for i in distances[:7]:
        # fetch the movie poster
        movie_id = ndf.iloc[i[0]].MAL_ID
        recommend_poster.append(fetch_poster(movie_id))
        recommendations.append(ndf.iloc[i[0]].Name)
    # rec_incat = pd.DataFrame(anime_list[anime_list['Name'].str.contains('^' + anime + '.*') == True].Name)
    return recommendations, recommend_poster


# Title of website
st.title('Find your Favorite Anime')

# sort
a_name = ndf['Name'].sort_values()

# select anime from our list or search
selection = st.selectbox(
    'Select or search for your favorite anime',
    a_name)  # see documentation for more info
similar_names = ndf[ndf['Name'].str.contains('^' + selection + '.*') == True]

if st.button('Show animes with similar names'):
    if len(similar_names) > 1:
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            # st.text(similar_names.iloc[0].Name)
            st.image(fetch_poster(similar_names.iloc[0].MAL_ID), width=128,
                     caption=similar_names.iloc[0].Name)
        with col2:
            # st.text(similar_names.iloc[1].Name)
            st.image(fetch_poster(similar_names.iloc[1].MAL_ID), width=128,
                     caption=similar_names.iloc[1].Name)
        with col3:
            # st.text(similar_names.iloc[2].Name)
            st.image(fetch_poster(similar_names.iloc[2].MAL_ID), width=128,
                     caption=similar_names.iloc[2].Name)
        with col4:
            # st.text(similar_names.iloc[3].Name)
            st.image(fetch_poster(similar_names.iloc[3].MAL_ID), width=128,
                     caption=similar_names.iloc[3].Name)
        with col5:
            # st.text(similar_names.iloc[4].Name)
            st.image(fetch_poster(similar_names.iloc[4].MAL_ID), width=128,
                     caption=similar_names.iloc[4].Name)
    else:
        st.error('No animes with similar names')

if st.button('Show Recommendation'):
    similarity = pickle.load(open('similarity2.pkl', 'rb'))
    #
    # ps = PorterStemmer()
    #
    #
    # def stem(text):
    #     y = []
    #
    #     for i in text.split():
    #         y.append(ps.stem(i))
    #     return ' '.join(y)
    #
    #
    # ndf.tags = ndf.tags.apply(stem)
    #
    # from sklearn.feature_extraction.text import CountVectorizer
    #
    # cv = CountVectorizer(max_features=5000, stop_words="english")
    #
    # vectors = cv.fit_transform(ndf['tags']).toarray()
    #
    # from sklearn.metrics.pairwise import cosine_similarity

    # similarity2 = cosine_similarity(vectors)
    recommendations, recommend_poster = recommend(selection)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        # st.text(recommendations[0])
        st.image(recommend_poster[0], width=128, caption=recommendations[0])
    with col2:
        # st.text(recommendations[1])
        st.image(recommend_poster[1], width=128, caption=recommendations[1])

    with col3:
        # st.text(recommendations[2])
        st.image(recommend_poster[2], width=128, caption=recommendations[2])
    with col4:
        # st.text(recommendations[3])
        st.image(recommend_poster[3], width=128, caption=recommendations[3])
    with col5:
        # st.text(recommendations[4])
        st.image(recommend_poster[4], width=128, caption=recommendations[4])
