# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np

# Set up matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.widgets import Slider, RangeSlider, CheckButtons
import matplotlib

from astropy.io import fits
import cv2
from PIL import Image
import sys

input_file = sys.argv[1]

# %%
#hdu_list = fits.open(input_file)
#hdu_list.info()


# %%
#image_data = hdu_list[0].data


# %%
#print(type(image_data))
#print(image_data.shape)


# %%
#hdu_list.close()


# %%
image_data = fits.getdata(input_file)
#print(type(image_data))
#print(image_data.shape)


# %%
meanIm = np.mean(image_data)
stdIm = np.std(image_data)
# ~ norm = LogNorm(vmin=meanIm-6*stdIm, vmax=meanIm+10*stdIm)
norm = Normalize(vmin=meanIm-6*stdIm, vmax=meanIm+10*stdIm)

thresh = np.percentile(image_data, 99.975)
# %%
#print('Min:', np.min(image_data))
#print('Max:', np.max(image_data))
#print('Mean:', np.mean(image_data))
#print('Stdev:', np.std(image_data))

# %%
fig, ax = plt.subplots(1, 1, figsize=(15, 15))
plt.subplots_adjust(left=0.1, bottom=0.25)
#divider = make_axes_locatable(ax)
#cax = divider.append_axes('right', size='5%', pad=0.05)

im1 = ax.imshow(image_data, cmap='gray')
#plt.colorbar(im1, cax=cax, orientation='vertical')

axcolor = 'w'
axminmax = plt.axes([0.03, 0.1, 0.03, 0.8], facecolor=axcolor)
rax = plt.axes([0.87, 0.4, 0.1, 0.1])

#s = RangeSlider(axminmax, 'Min/Max', 0, thresh, color='k', orientation='vertical')


class WidgetClickProcessor(object):

    def __init__(self, axes_button, img, norm, axes_slider):
        self.slider = RangeSlider(axes_slider, 'Min/Max', 0, thresh, color='k', orientation='vertical')
        self.img = img

        meanIm = np.mean(img)
        stdIm = np.std(img)
        self.vmin = meanIm-6*stdIm
        self.vmax = meanIm+10*stdIm
        self.norm = Normalize(vmin=self.vmin, vmax=self.vmax)

        self.button = CheckButtons(axes_button, ['Normalize'])
        self.button.on_clicked(self.process)
        self.slider.on_changed(self.update)
        ax.imshow(self.img, cmap='gray', vmin=self.slider.val[0], vmax=self.slider.val[1])

    def process(self, event):
        if self.button.get_status()[0] == False:
            ax.imshow(self.img, cmap='gray', vmin=self.slider.val[0], vmax=self.slider.val[1])
        else:
            ax.imshow(self.img, cmap='gray', norm=self.norm)
            self.slider.set_val((self.vmin, self.vmax))
        plt.draw()

    def update(self, val):
        ax.imshow(self.img, cmap='gray', vmin=self.slider.val[0], vmax=self.slider.val[1])
        #plt.colorbar(im1, cax=cax, orientation='vertical')
        plt.draw()

widgets = WidgetClickProcessor(rax, image_data, norm, axminmax)

print(matplotlib.matplotlib_fname())
fig.tight_layout()
plt.show()

