from google_images_download import google_images_download
from webdriver_manager.chrome import ChromeDriverManager

response = google_images_download.googleimagesdownload()

arg = {"keywords": "fire cctv, 화재 cctv, 화재 사고, fire accident, fire accident cctv",
       "limit": 15000,
       "print_urls": True,
       "possible_format": "jpg, png",
       "output_directory": "/media/daton/D6A88B27A88B0569/dataset/fire detection",
       "chromedriver": ChromeDriverManager().install()}
paths = response.download(arg)

print(paths)