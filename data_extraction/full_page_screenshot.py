import time
from typing import Any

from pydantic import BaseModel
from selenium.webdriver import Chrome, ChromeOptions
from selenium.webdriver.common.by import By


class PageScreenshot(BaseModel):
    """PageScreenshot makes a screenshot of a page by url."""

    options: Any
    driver: Any

    def S(self, prop: str): return self.driver.execute_script(
        'return document.body.parentNode.scroll'+prop)

    def get_screenshot(self, url: str, id: int) -> None:
        try:
            self.driver.get(url)

            self.driver.set_window_size(self.S('Width'), self.S('Height'))
            page = self.driver.find_element(By.TAG_NAME, 'body').screenshot(
                f'web_screenshot_{id}.png')

            # self.driver.quit()
        except SyntaxError:
            pass

    def get_screenshots_by_url(self, urls: list) -> None:

        for id, url in enumerate(urls):
            self.get_screenshot(url, id)

        self.driver.quit()


if __name__ == "__main__":
    options = ChromeOptions()
    options.headless = True
    driver = Chrome(options=options)
    page_screenshot = PageScreenshot(options=options, driver=driver)
    urls = ['https://uaserials.pro/715-mentalist-sezon-1.html', 'https://pythonbasics.org/selenium-screenshot/',
            'https://pythonexamples.org/python-selenium-find-element-by-tag-name/']
    page_screenshot.get_screenshots_by_url(urls)
