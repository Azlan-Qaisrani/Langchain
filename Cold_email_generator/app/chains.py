# -*- coding: utf-8 -*-
# email_chain.py

import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException

load_dotenv()  # load GROQ_API_KEY from .env


class Chain:
    def __init__(self):
        # Use environment variable, do not pass grok_api_key
        self.llm = ChatGroq(
            temperature=0,
            model="openai/gpt-oss-120b"
        )

    def extract_jobs(self, cleaned_text: str) -> list:
        """Extract job postings from website text into JSON."""
        prompt_extract = PromptTemplate.from_template("""
### SCRAPED TEXT FROM WEBSITE:
{page_data}
### INSTRUCTION:
The scraped text is from the career's page of a website.
Your job is to extract the job postings and return them in JSON format containing the following keys: `role`, `experience`, `skills` and `description`.
Only return the valid JSON.
### VALID JSON (NO PREAMBLE):
""")
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={"page_data": cleaned_text})

        # Ensure content is string before parsing
        content_str = json.dumps(res.content) if isinstance(
            res.content, (list, dict)) else str(res.content)

        json_parser = JsonOutputParser()
        try:
            jobs = json_parser.parse(content_str)
        except OutputParserException as e:
            raise OutputParserException(f"Failed to parse jobs: {str(e)}")

        return jobs if isinstance(jobs, list) else [jobs]

    def write_mail(self, job: dict, links: list) -> str:
        """Generate cold email for a given job posting."""
        prompt_email = PromptTemplate.from_template("""
### JOB DESCRIPTION:
{job_description}

### INSTRUCTION:
You are Azlan, a business development executive at DevsInc. DevsInc is an AI & Software Consulting company dedicated to facilitating
the seamless integration of business processes through automated tools. 
Over our experience, we have empowered numerous enterprises with tailored solutions, fostering scalability, 
process optimization, cost reduction, and heightened overall efficiency. 
Your job is to write a cold email to the client regarding the job mentioned above describing the capability of DevsInc 
in fulfilling their needs.
Also add the most relevant ones from the following links to showcase DevsInc's portfolio: {link_list}
Remember you are Azlan, BDE at DevsInc. 
Include your contact information at the end: Email: azlanmuhammd675@gmail.com, Phone: 03186086867
Do not provide a preamble.

### EMAIL (NO PREAMBLE):
""")
        chain_email = prompt_email | self.llm

        # Ensure links are a single string
        link_str = ", ".join(links) if isinstance(links, list) else str(links)

        res = chain_email.invoke({
            "job_description": json.dumps(job) if isinstance(job, (dict, list)) else str(job),
            "link_list": link_str
        })

        # Return string content
        return res.content if isinstance(res.content, str) else str(res.content)


# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    # Test: make sure GROQ_API_KEY is set in your .env
    print("GROQ_API_KEY:", os.getenv("GROQ_API_KEY"))
