# TODO: Make sure all keys are in gpt response and retry if needed
# TODO: Parse PDF and get GPT summaries

import json
import os
import requests
import xml.etree.ElementTree as ET

from bs4 import BeautifulSoup
from collections import defaultdict
from datetime import datetime
from openai import OpenAI
from tenacity import retry, stop_after_attempt, retry_if_exception_type
from tqdm import tqdm


API_KEY = os.environ["OPENAI_API_KEY"]

MANIFESTO = """
I am interested in reading machine learning research papers on 3 main high-level topics, which I outline below.
For each topic, I include a list of subtopics of interest.

1) Time series and deep learning (tag: time-series)
    - Deep learning models for time series
    - Foundation models for time series
    - Transformers for time series

2) Causality and machine learning (tag: causality)
    - Causal representation learning
    - Causal discovery

3) Agents based on large-language models (tag: agents)
    - Agents that interact with software

"""


class Paper:
    def __init__(self, title, authors, description, link, guid, category):
        self.title = title
        self.authors = authors
        self.summary = description
        self.link = link
        self.guid = guid
        self.category = category

        self.__interest_analysis = None

    # Add a decorator to retry 5 times max
    @retry(
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type(Exception),
    )
    @property
    def _interest_analysis(self):
        if self.__interest_analysis is None:
            completion = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an assistant helping a user find research papers that match their interests.",
                    },
                    {
                        "role": "user",
                        "content": MANIFESTO
                        + f"""
                Here are the title and description of a paper:
                Title: {self.title}
                Description: {self.summary}

                Give it a score from 1 to 5, where 1 means "not relevant" and 5 means "very relevant".
                Also, provide the tag from my interest to which it corresponds and the subtopic that is most relevant.
                Answer in json format.

                Example:
                    {{
                        "score": 5,
                        "tag": "time-series",
                        "subtopic": "Deep learning models for time series"
                    }}
                """,
                    },
                ],
            )
            self.__interest_analysis = json.loads(completion.choices[0].message.content)

            # Try accessing the keys in the response to trigger a retry if needed
            self.__interest_analysis["score"]
            self.__interest_analysis["tag"]
            self.__interest_analysis["subtopic"]

        return self.__interest_analysis

    @property
    def interest_score(self):
        return self._interest_analysis["score"]

    @property
    def interest_tag(self):
        return self._interest_analysis["tag"]

    @property
    def interest_topic(self):
        return self._interest_analysis["subtopic"]

    def to_html(self):
        """
        Return a div with the paper information styled beautifully.

        """
        return f"""
        <div class="paper-box">
            <h3>{self.title}</h3>
            <p><strong>Authors:</strong> {self.authors}</p>
            <p><strong>Summary:</strong> {self.summary}</p>
            <p><strong>Link:</strong> <a href="{self.link}" target="_blank">{self.link}</a></p>
            <p><strong>Category:</strong> {self.category}</p>
            <p><strong>Interest score:</strong> {self.interest_score}</p>
            <p><strong>Interest tag:</strong> {self.interest_tag} (topic: {self.interest_topic})</p>
        </div>
        """

    def __str__(self):
        return f"{self.title} by {self.authors}"


def html_report(filename, interesting):
    """
    Generate an HTML report with the interesting papers, including styles,
    with capitalized tag titles and papers sorted by decreasing interest score.
    Include the current date and time of generation.
    """
    current_date = datetime.now().strftime("%B %d, %Y")
    current_time = datetime.now().strftime("%H:%M:%S")

    with open(filename, "w") as f:
        # Adding the opening HTML structure along with style and title.
        f.write(
            f"""
        <html>
        <head>
            <title>Report Generated on {current_date}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .paper-box {{
                    background-color: #f0f0f0;
                    margin-bottom: 20px;
                    padding: 15px;
                    border-radius: 5px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                h1 {{ text-align: center; }}
                h2 {{
                    color: #333;
                    border-bottom: 2px solid #666;
                }}
                a {{ color: #337ab7; text-decoration: none; }}
                a:hover {{ text-decoration: underline; }}
                h3 {{ color: #337ab7; }}
                .timestamp {{ text-align: center; font-size: small; margin-top: 40px; }}
            </style>
        </head>
        <body>
        <h1>Report for {current_date}</h1>
        """
        )

        # Sort papers in each tag group by decreasing interest score
        # and capitalize tag titles.
        for tag, papers in interesting.items():
            capitalized_tag = tag.capitalize()
            sorted_papers = sorted(
                papers, key=lambda paper: paper.interest_score, reverse=True
            )

            f.write(f"<h2>{capitalized_tag}</h2>")
            f.write("<div class='papers-container'>")
            for paper in sorted_papers:
                f.write(paper.to_html())
            f.write("</div>")

        # Add timestamp at the bottom
        f.write(
            f"<div class='timestamp'>Report generated on {current_date} at {current_time}</div>"
        )

        f.write("</body></html>")


def get_papers():
    # Make a request to the ArXiv RSS feed
    response = requests.get("https://rss.arxiv.org/rss/cs.lg")
    response.raise_for_status()

    # Parse the XML document
    root = ET.fromstring(response.content)
    papers = []
    for item in root.findall(".//item"):
        papers.append(
            Paper(
                title=item.find("title").text,
                authors=item.find("{http://purl.org/dc/elements/1.1/}creator").text,
                description=item.find("description").text,
                link=item.find("link").text,
                guid=item.find("guid").text,
                category=item.find("category").text,
            )
        )
    return papers


openai_client = OpenAI(api_key=API_KEY)
papers = get_papers()

interesting = defaultdict(list)
for paper in tqdm(papers):
    if paper.interest_score > 3:
        interesting[paper.interest_tag].append(paper)

# Generate an HTML report. Store it in a file tagged with the date
html_report(f"papers_{datetime.now().strftime('%Y_%m_%d')}.html", interesting)
