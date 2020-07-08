import io
from typing import Optional, NoReturn

from wand.color import Color
from wand.drawing import Drawing
from wand.image import Image


class SnippetImageMaker:
    def __init__(self, width: int = 720, height: int = 720, snippet_height: int = 180) -> NoReturn:
        self.width, self.height, self.snippet_height = width, height, snippet_height
        self.img: Optional[Image] = None
        self.pic: Optional[Image] = None

    def create_img(self, pic: io.BytesIO, color: Color = Color('WHITE')) -> NoReturn:
        """Create background and load picture."""

        self.img = Image(width=self.width, height=self.height, background=color)
        self.pic = self.prepare_picture(Image(file=pic))
        self.img.composite(self.pic)

    def prepare_picture(self, pic: Image) -> Image:
        """Resize and/or cropping of picture."""

        # height of picture on snippet image
        if pic.width == self.width and pic.height == self.pic_height:
            return pic
        img_pic_h = self.height - self.snippet_height
        if pic.width > pic.height:
            pic.resize(self.width, img_pic_h)
        else:
            pic.resize(width=self.width)
            if pic.height < img_pic_h:
                pic.resize(height=img_pic_h)
            else:
                pic.crop(top=int((pic.height - img_pic_h) / 2), height=img_pic_h)
        return pic

    def add_text_block(self, text: str, font: str, font_size: int, font_color: str,
                       x: int, y: int) -> int:
        """Add text to self.img."""

        with Drawing() as draw:
            draw.font = font
            draw.font_size = font_size
            draw.fill_color = font_color
            draw.text(x, y, text)
            draw(self.img)
            return int(draw.get_font_metrics(image=self.img, text=text).text_width)

    def add_svg_block(self, svg_string: str, x: int, y: int, encoding: str = "utf-8") -> int:
        """Add svg block to self.img."""

        svg = io.BytesIO(bytes(svg_string, encoding=encoding))
        with Image(file=svg, format="svg") as svg_img:
            self.img.composite(svg_img, x, y)
            return svg_img.width

    def save_img(self, filename: str, img_format: str):
        """Save img with given format."""

        self.img.format = img_format
        with open(filename, "wb") as f:
            f.write(self.img.make_blob())

    def get_img_blob(self, img_format: str):
        """Get image blob with given format."""

        self.img.format = img_format
        return self.img.make_blob()

    def get_text_width(self, text: str, text_size: int, text_font: str):
        """Get text width after overlay on image."""

        with Drawing() as draw:
            draw.font = text_font
            draw.font_size = text_size
            return int(draw.get_font_metrics(image=self.img, text=text).text_width)

    @property
    def pic_height(self):
        return self.height - self.snippet_height
