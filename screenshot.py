"""
This program takes a screenshot and prints the result as a matrix of pixels. 
"""


from PIL import ImageGrab
import numpy as np
 
 
def screenGrab():
    im = ImageGrab.grab()
    pixels = list(im.getdata())
    width, height = im.size
    pixels = np.array([pixels[i * width:(i + 1) * width] for i in range(height)])
    print(pixels)
    
def main():
    screenGrab()
 
if __name__ == '__main__':
    main()