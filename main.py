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
For each topic, I include a list of subtopics that I care about. Help me find papers that are most relevant to my
interests. Note that I'm mostly interest in research papers the propose new methods more than applications.

1) Time series and deep learning. Mostly forecasting. (tag: time-series)
    - New deep learning methods for time series
    - New foundation models for time series
    - Datasets to train foundation models for time series
    - New multimodal deep learning models for time series
    - New transformer-like models for time series

2) Causality and machine learning (tag: causality)
    - Causal representation learning
    - Causal discovery
    - Using large language models in causal discovery

3) Agents based on large-language models (tag: llm-agents)
    - Using large language models to control software
    - Using large language models to control web browsers
    - Computer automation using large language models

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

    @property
    @retry(
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type(Exception),
    )
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
                I won't even look at the papers with a score lower than 3.
                Also, provide the tag from my interest to which it corresponds, along with a short justification.
                Answer in json format.

                Example:
                    {{
                        "score": 5,
                        "tag": "time-series",
                        "justification": "You should read this paper because it proposes a new multimodal approach to time series forecasting."
                    }}
                """,
                    },
                ],
            )
            self.__interest_analysis = json.loads(completion.choices[0].message.content)

            # Try accessing the keys in the response to trigger a retry if needed
            self.__interest_analysis["score"]
            self.__interest_analysis["tag"]
            self.__interest_analysis["justification"]

        return self.__interest_analysis

    @property
    def interest_score(self):
        return self._interest_analysis["score"]

    @property
    def interest_tag(self):
        return self._interest_analysis["tag"]

    @property
    def interest_justification(self):
        return self._interest_analysis["justification"]

    def to_html(self):
        """
        Return a div with the paper information and the main figure thumbnail styled beautifully.
        """
        # Get the main figure URL or None
        main_figure_url = self.main_figure
        main_figure_html = (f'<a href="{main_figure_url}" target="_blank">'
                            f'<img src="{main_figure_url}" class="paper-figure" alt="Main Figure"></a>'
                            if main_figure_url else '')

        return f"""
        <div class="paper-box">
            <h3><a href="{self.link}" target="_blank">{self.title}</a></h3>
            {main_figure_html}
            <p><strong>Authors:</strong> {self.authors}</p>
            <p><strong>Summary:</strong> {self.summary}</p>
            <p><strong>Link:</strong> <a href="{self.link}">{self.link}</a></p>
            <p><strong>Category:</strong> {self.category}</p>
            <p><strong>Interest score:</strong> {self.interest_score}</p>
            <p><strong>Interest tag:</strong> {self.interest_tag}</p>
            <p><strong>Interest justification:</strong> {self.interest_justification}</p>
        </div>
        """

    @property
    def main_figure(self):
        """
        Return the main figure of the paper as a URL.

        """
        url = f"https://arxiv.org/html/{self.guid.split(':')[-1]}"
        response = requests.get(url)
        if response.status_code != 200:
            return None
        soup = BeautifulSoup(response.content, "html.parser")
        img = soup.find("img", class_="ltx_graphics")
        return url + "/" + img.get("src", None) if img else None

    def __str__(self):
        return f"{self.title} by {self.authors}"


def html_report(filename, interesting):
    """
    Generate an HTML report with the interesting papers, including styles and JavaScript,
    with capitalized tag titles and papers sorted by decreasing interest score.
    Include the current date and time of generation. Each section is collapsible.
    """
    current_date = datetime.now().strftime("%B %d, %Y")
    current_time = datetime.now().strftime("%H:%M:%S")

    with open(filename, "w") as f:
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
                        cursor: pointer;
                        color: #333;
                        border-bottom: 2px solid #666;
                    }}
                    a {{ color: #337ab7; text-decoration: none; }}
                    a:hover {{ text-decoration: underline; }}
                    h3 {{ color: #337ab7; }}
                    .timestamp {{ text-align: center; font-size: small; margin-top: 40px; }}
                    .paper-figure {{
                        max-width: 200px;
                        border: 1px solid #ddd;
                        border-radius: 4px;
                        padding: 5px;
                        margin-top: 10px;
                    }}
                    .papers-container {{ display: block; padding: 0 18px; }}
                </style>
            </head>
            <body>
            <h1>Report for {current_date}</h1>
            <script>
                function toggleSection(id) {{
                    var x = document.getElementById(id);
                    if (x.style.display === "none") {{
                        x.style.display = "block";
                    }} else {{
                        x.style.display = "none";
                    }}
                }}
            </script>
            """
        )

        for tag, papers in interesting.items():
            capitalized_tag = tag.capitalize()
            sorted_papers = sorted(papers, key=lambda paper: paper.interest_score, reverse=True)
            section_id = f"section_{tag}"

            f.write(f"<h2 onclick=\"toggleSection('{section_id}')\">{capitalized_tag}</h2>")
            f.write(f"<div id='{section_id}' class='papers-container'>")
            for paper in sorted_papers:
                f.write(paper.to_html())
            f.write("</div>")

        f.write(f"<div class='timestamp'>Report generated on {current_date} at {current_time}</div>")
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
