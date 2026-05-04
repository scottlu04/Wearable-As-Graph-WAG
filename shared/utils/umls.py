import requests


import html
import re
from typing import Any


# def umls_search(input: Any) -> str:
#     """Clean an input string by removing HTML escapes, control characters, and other unwanted characters."""
#     # If we get non-string input, just give it back
#     if not isinstance(input, str):
#         return input

#     result = html.unescape(input.strip())
#     # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python
#     return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)



class umls:
    def __init__(self, apikey, version="current"):
        self.apikey = apikey
        self.version = version
        self.search_url = f"https://uts-ws.nlm.nih.gov/rest/search/{version}"
        self.content_url = f"https://uts-ws.nlm.nih.gov/rest/content/{version}"
        self.content_suffix = "/CUI/{}/{}?apiKey={}"

    def search_cui(self, query):
        cui_results = []

        try:
            page = 1
            size = 1
            query = {"string": query, "apiKey": self.apikey, "pageNumber": page, "pageSize": size}
            r = requests.get(self.search_url, params=query)
            r.raise_for_status()
            print(r.url)
            r.encoding = 'utf-8'
            outputs = r.json()

            items = outputs["result"]["results"]

            if len(items) == 0:
                print("No results found.\n")

            for result in items:
                cui_results.append((result["ui"], result["name"]))

        except Exception as except_error:
            print(except_error)

        return cui_results

    def get_definitions(self, cui):
        try:
            suffix = self.content_suffix.format(cui, "definitions", self.apikey)
            r = requests.get(self.content_url + suffix)
            r.raise_for_status()
            r.encoding = "utf-8"
            outputs = r.json()

            return outputs["result"]
        except Exception as except_error:
            print(except_error)

    def get_relations(self, cui, pages=20):
        all_relations = []

        try:
            for page in range(1, pages + 1):
                suffix = self.content_suffix.format(cui, "relations", self.apikey) + f"&pageNumber={page}"
                r = requests.get(self.content_url + suffix)
                r.raise_for_status()
                r.encoding = "utf-8"
                outputs = r.json()

                page_relations = outputs.get("result", [])
                all_relations.extend(page_relations)

        except Exception as except_error:
            print(except_error)

        return all_relations