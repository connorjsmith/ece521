from PIL import Image

image_w_h = 396
width = image_w_h*10
height = image_w_h*10
new_im = Image.new('RGBA', (width, height))

x_offset = 0
for irow in range(0, 10): 
    y_offset = 0
    for icol in range(0, 10): 
        img = Image.open("100_dropout{}.png".format(irow*10+icol))
        new_im.paste(img, (x_offset, y_offset))
        y_offset += image_w_h
    x_offset += image_w_h


new_im.save('100_dropout_combined.png')
