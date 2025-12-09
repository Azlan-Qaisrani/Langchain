# main.py

import streamlit as st
from langchain_community.document_loaders import WebBaseLoader

# your Chain class for job extraction & email generation
from chains import Chain
from portfolio import Portfolio  # your Portfolio class
from utils import clean_text    # function to clean scraped text


def create_streamlit_app(chain, portfolio, clean_text_func):
    st.title("📧 Cold Mail Generator")

    url_input = st.text_input(
        "Enter a job posting URL:",
        value="https://www.mustakbil.com/jobs/pakistan/information-technology"
    )
    submit_button = st.button("Generate Emails")

    if submit_button:
        if not url_input:
            st.warning("Please enter a valid URL.")
            return

        try:
            # Load and clean page content
            loader = WebBaseLoader([url_input])
            page_data = loader.load()
            if not page_data:
                st.warning(
                    "Failed to load page content. Make sure the URL is correct and accessible.")
                return

            cleaned_text = clean_text(page_data.pop().page_content)

            # Load portfolio data
            portfolio.load_portfolio()

            # Extract jobs
            jobs = chain.extract_jobs(cleaned_text)
            if not jobs:
                st.info("No jobs were extracted from the page.")
                return

            # Generate emails for each job
            for job in jobs:
                skills = job.get("skills", [])
                links = portfolio.query_links(skills)
                email = chain.write_mail(job, links)

                st.subheader(f"Email for Role: {job.get('role', 'N/A')}")
                st.code(email, language='markdown')
                st.markdown("---")

        except Exception as e:
            st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    # Initialize your classes
    chain = Chain()
    portfolio = Portfolio()

    # Streamlit page config
    st.set_page_config(
        layout="wide",
        page_title="Cold Email Generator",
        page_icon="📧"
    )

    # Run the Streamlit app
    create_streamlit_app(chain, portfolio, clean_text)
