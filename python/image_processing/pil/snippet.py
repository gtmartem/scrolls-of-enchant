import io
from typing import NoReturn, Optional, Tuple

import cairosvg
from PIL import Image, ImageFont

from PIL import ImageDraw


class SnippetImageMaker:
    def __init__(self, width: int = 720, height: int = 720, snippet_height: int = 180) -> NoReturn:
        self.width, self.height, self.snippet_height = width, height, snippet_height
        self.img: Optional[Image] = None
        self.pic: Optional[Image] = None

    def create_img(self, pic: io.BytesIO, color: Tuple = (255, 255, 255)) -> NoReturn:
        """Create background and load picture."""

        self.img = Image.new('RGB', (self.width, self.height), color)
        self.pic = self.prepare_picture(Image.open(pic))
        self.img.paste(self.pic, (0, 0))

    def prepare_picture(self, pic: Image) -> Image:
        """Resize and/or cropping of picture."""

        # height of picture on snippet image
        if pic.width == self.width and pic.height == self.pic_height:
            return pic
        img_pic_h = self.height - self.snippet_height
        if pic.width > pic.height:
            updated_pic = pic.resize((self.width, img_pic_h))
        else:
            updated_pic = pic.resize((self.width, pic.height))
            if pic.height < img_pic_h:
                updated_pic = updated_pic.resize((updated_pic.width, img_pic_h))
            else:
                # The crop method from the Image module takes four coordinates as input.
                # The right can also be represented as (left+width)
                # and lower can be represented as (upper+height).
                vertical_shift = int((updated_pic.height - img_pic_h) / 2)
                (left, upper, right, lower) = (0, vertical_shift, 0, vertical_shift)
                updated_pic = updated_pic.crop((left, upper, right, lower))
        return updated_pic

    def add_text_block(self, text: str, font: str, font_size: int,
                       x: int, y: int, font_color: tuple = (0, 0, 0),) -> int:
        """Add text to self.img."""

        d_ctx = ImageDraw.Draw(self.img)
        fnt = ImageFont.truetype(font, font_size)
        d_ctx.text((x, y), text, font=fnt, fill=font_color)
        return d_ctx.textsize(text=text, font=fnt)

    def add_svg_block(self, svg_string: str, x: int, y: int, encoding: str = "utf-8") -> int:
        """Add svg block to self.img."""

        out = io.BytesIO()
        svg = io.BytesIO(bytes(svg_string, encoding=encoding))
        cairosvg.svg2png(file_obj=svg, write_to=out)
        svg_b = Image.open(out)
        self.img.paste(svg_b, (x, y), svg_b)
        return svg_b.size[0]

    def save_img(self, filename: str, img_format: str):
        """Save img with given format."""

        bts_f = io.BytesIO()
        self.img.save(bts_f, format=img_format)
        with open(filename, "wb") as f:
            f.write(bts_f.getvalue())

    def get_img_blob(self, img_format: str):
        """Get image blob with given format."""

        bts_f = io.BytesIO()
        self.img.save(bts_f, format=img_format)
        return bts_f.getvalue()

    def get_text_width(self, text: str, text_size: int, text_font: str):
        """Get text width after overlay on image."""

        d_ctx = ImageDraw.Draw(self.img)
        fnt = ImageFont.truetype(text_font, text_size)
        return d_ctx.textsize(text, font=fnt)[0]

    @property
    def pic_height(self):
        return self.height - self.snippet_height
