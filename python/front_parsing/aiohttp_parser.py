import asyncio
from typing import Optional

import aiohttp
from bs4 import BeautifulSoup


FOOTER_DIV_CLASS = "_2qozQ"
FOOTER_LINKS_CLASS = "_3A_Wi"
URL = "yandex.ru"


class FrontParser:

    session: Optional[aiohttp.ClientSession] = None

    def __init__(self):
        pass

    async def setup(self):
        self.session = aiohttp.ClientSession()

    async def parse(self, url: str):
        page = await self.session.get(url)
        if page.status == 200:
            soup = BeautifulSoup(await page.text(), 'html.parser')
            footer_block = soup.find("div", attrs={'class': '_2qozQ'})
            links = footer_block.select("a", attrs={'class': '_3A_Wi'})


async def main(parser: FrontParser, url: str):
    await parser.setup()
    await parser.parse(url)


if __name__ == '__main__':
    asyncio.run(main(FrontParser(), URL))

