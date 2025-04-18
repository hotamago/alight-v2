import qrcode
import numpy as np

class QRCodeB:
    _qr = None
    def __init__(self, version = 1, error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=5, border=2):
        """__init__ object qr function for bincase project

        Args:
            version (int, optional): The version parameter is an integer from 1 to 40 that controls the size of the QR Code (the smallest, version 1, is a 21x21 matrix). Defaults to 1.
            error_correction (qrcode.constants, optional): The error_correction parameter controls the error correction used for the QR Code. Detail: https://pypi.org/project/qrcode/. Defaults to qrcode.constants.ERROR_CORRECT_L.
            box_size (int, optional): The box_size parameter controls how many pixels each “box” of the QR code is. Defaults to 5.
            border (int, optional): The border parameter controls how many boxes thick the border should be. Defaults to 2.
        """
        self._version = version
        self._qr = qrcode.QRCode(
            version=version,
            error_correction=error_correction,
            box_size=box_size,
            border=border,
        )
    def make(self, text, fill_color="black", back_color="white"):
        """Create a QR code

        Args:
            text (str): Data of QR code
            fill_color (str, optional): Defaults to "black".
            back_color (str, optional): Defaults to "white".
            fill_color and back_color can change the background and the painting color of the QR, when using the default image factory. Both parameters accept RGB color tuples

        Returns:
            numpy.array: Image description by array
        """
        self._qr.clear()
        self._qr.add_data(text)
        self._qr.make(fit=True)

        pil_img = self._qr.make_image(fill_color=fill_color, back_color=back_color).convert("L")
        return np.asarray(pil_img, dtype=np.uint8)
    def add_corners_qr(self, img, core_text = "bin"):
        """Add 4 qr code to corners of given image

        Args:
            img (numpy.array): Init image
            core_text (str, optional): Vertify data text corners. Defaults to "bin".

        Returns:
            numpy.array: Image with 4 qrcode in corners
        """
        img_tl, img_tr, img_bl, img_br = self.make(core_text + '-tl'), self.make(core_text + '-tr'), self.make(core_text + '-bl'), self.make(core_text + '-br')
        img[:img_tl.shape[0], :img_tl.shape[1]] = img_tl
        img[:img_tl.shape[0], img.shape[1] - img_tl.shape[1]:] = img_tr
        img[img.shape[0] - img_tl.shape[0]:, :img_tl.shape[1]] = img_bl
        img[img.shape[0] - img_tl.shape[0]:, img.shape[1] - img_tl.shape[1]:] = img_br
        return img
    def given_image_corners_qr(self, size, core_text = "bin"):
        """Create a white backgroud image and add 4 qrcode to corners

        Args:
            size (tuple): size of image
            core_text (str, optional): Vertify data text corners. Defaults to "bin".

        Returns:
            numpy.array: Image with 4 qrcode in corners and white backgroud
        """
        img = np.full((size[1], size[0]), 255, dtype=np.uint8)
        img = self.add_corners_qr(img, core_text)
        return img