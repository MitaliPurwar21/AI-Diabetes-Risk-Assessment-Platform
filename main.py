import streamlit as st
from Tabs import diagnosis, home, result, kc, talk2doc

# Configure the app
st.set_page_config(
    page_title='AI-Powered Diabetes Risk Assessment Platform',
    page_icon='🥯',
    layout='wide',
    initial_sidebar_state='auto'
)

# Tab mapping
Tabs = {
    "Home": home,
    "Ask Queries": talk2doc,
    "Diagnosis": diagnosis,
    "Result": result,
    "Knowledge Center": kc
}

# Sidebar navigation
st.sidebar.title('Navigation')
page = st.sidebar.radio("Page", list(Tabs.keys()))
st.sidebar.info('Made by Mitali')

# Run selected tab
if page == "Diagnosis":
    # diagnosis.py will handle model loading and user input
    Tabs[page].app()
else:
    Tabs[page].app()
