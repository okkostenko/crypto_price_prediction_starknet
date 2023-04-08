import json
import os
import re
from io import BytesIO

import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
from PIL import Image
from pydantic import BaseModel

DATA_PATH = './data'


class BehanceParser(BaseModel):

    """BeganceParser gets images of a full designs of a landing pages from a website called Behance, which is one of the most popular social media platforms for creative people."""

    base_url: str = 'https://www.behance.net/search/projects?search=landing+page'
    atag_title: str = 'Link to project'

    def check_image_format(self, url: str) -> bool:
        return (url.endswith('.jpeg') or url.endswith('.jpg') or url.endswith('.png'))

    def projects_url(self) -> list:
        response = requests.get(self.base_url).text
        soup = BeautifulSoup(response, 'html.parser')

        # hrefs = []
        # for _ in range(5):
        #     hrefs.append(soup.find('a', {'title': self.atag_title})['href'])

        # Uncomment when need to use for all of the links on the page
        atags = soup.find_all('a', {'title': self.atag_title})
        hrefs = [atag['href'] for atag in atags]
        return hrefs

    def design_url(self, hrefs: list) -> list:
        design_urls = []
        for url in hrefs:
            response = requests.get(url).text
            soup = BeautifulSoup(response, 'html.parser')
            image_links = [link['src'] for link in soup.find_all(
                'img', {"class": "ImageElement-image-SRv"}) if self.check_image_format(link['src'])]
            design_urls.append(image_links[-1])

        return design_urls

    def get_image(self, design_urls: list) -> None:
        for id, url in enumerate(design_urls):
            image = Image.open(
                BytesIO(requests.get(url).content)).convert('RGB')
            plt.imshow(image)
            image.save(os.path.join(DATA_PATH, f'behance_image_{id}.png'))


if __name__ == "__main__":
    parser = BehanceParser()
    # project_urls = parser.projects_url()
    project_urls = ['https://www.behance.net/gallery/165599187/Crypto-Landing-Page?tracking_source=search_projects%7Clanding+page', 'https://www.behance.net/gallery/166141745/NFT-Landing-Page-Design?tracking_source=search_projects%7Clanding+page',
                    'https://www.behance.net/gallery/166882821/NFT-Landing-Page-Design?tracking_source=search_projects%7Clanding+page', 'https://www.behance.net/gallery/158203647/Azuki_Website-Redesign-Landing-Page?tracking_source=search_projects%7Clanding+page']
    design_urls = parser.design_url(project_urls)
    parser.get_image(design_urls)
