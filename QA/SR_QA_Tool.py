import cv2, argparse
import numpy as np
from matplotlib import pyplot as plt
from skimage.measure import compare_ssim
from mpl_toolkits.mplot3d import Axes3D
import imquality.brisque as brisque
from PIL import Image
from matplotlib.ticker import MaxNLocator

ix, iy = -1, -1
mode = False
original, original_copy, target, target_copy, original_BRISQUE, target_BRISQUE = None, None, None, None, None, None
imshow_width = 960
imshow_height = 600

def SSIM(roi_source, roi_target):
    roi_source = cv2.cvtColor(roi_source, cv2.COLOR_BGR2GRAY)
    roi_target = cv2.cvtColor(roi_target, cv2.COLOR_BGR2GRAY)

    return compare_ssim(roi_source, roi_target, multichannel=True, full=True)

def fourierTransform(roi):
    img_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    f = np.fft.fft2(img_gray)
    fshift = np.fft.fftshift(f)
    abs_fft = np.abs(fshift)
    abs_fft = np.log(1 + abs_fft)

    magnitude_spectrum = 20 * abs_fft
    fft_aug_norm = cv2.normalize(abs_fft, 0, 255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)

    rows, cols = img_gray.shape
    crow, ccol = int(rows / 2), int(cols / 2)

    fshift[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0
    f_ishift = np.fft.ifftshift(fshift)
    HF = np.fft.ifft2(f_ishift)
    HF = np.abs(HF)

    return magnitude_spectrum, fft_aug_norm, HF, img_gray.shape

def onMouse(event,x,y,flag,param):
    global ix,iy,mode, original, original_copy, target, target_copy, original_BRISQUE, target_BRISQUE

    target_height, target_width, target_channel = target.shape
    rect_coord_multiple_width = target_width / imshow_width
    rect_coord_multiple_height = target_height / imshow_height

    if event == cv2.EVENT_LBUTTONDOWN:
        mode = True
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if mode:
            original = original_copy.copy()
            cv2.rectangle(original, (int(ix*rect_coord_multiple_width)-10, int(iy*rect_coord_multiple_height)-10),
                                    (int(x*rect_coord_multiple_width)+10, int(y*rect_coord_multiple_height)+10), (0, 0, 255), 10)
            cv2.resizeWindow('Image Comparison', imshow_width, imshow_height)
            cv2.imshow('Image Comparison', cv2.resize(original.copy(), (imshow_width, imshow_height)))

    elif event == cv2.EVENT_LBUTTONUP:
        mode = False
        if ix >= x or iy >= y:
            return

        roi_original = original[int(iy*rect_coord_multiple_height):int(y*rect_coord_multiple_height), int(ix*rect_coord_multiple_width):int(x*rect_coord_multiple_width)]
        roi_target = target[int(iy*rect_coord_multiple_height):int(y*rect_coord_multiple_height), int(ix*rect_coord_multiple_width):int(x*rect_coord_multiple_width)]
        magnitude_spectrum_original, fft_aug_norm_original, HF_original, (h_original, w_original) = fourierTransform(roi_original)
        magnitude_spectrum_target, fft_aug_norm_target, HF_target, (h_target, w_target) = fourierTransform(roi_target)
        score, diff = SSIM(roi_original, roi_target)

        # Set up a figure twice as tall as it is wide
        fig = plt.figure(figsize=plt.figaspect(0.8)) # 1.
        fig.suptitle('Image Analysis in Frequency Domain')
        # plt.tight_layout()

        # Plots
        ax = fig.add_subplot(3, 4, 1)
        ax.imshow(cv2.cvtColor(roi_original, cv2.COLOR_BGR2RGB))
        ax.set_xlabel('Width')
        ax.set_ylabel('Height')
        ax.set_title('(a) Bicubic ROI Image')

        ax = fig.add_subplot(3, 4, 2)
        ax.imshow(HF_original, cmap='gray')
        ax.set_xlabel('Width')
        ax.set_ylabel('Height')
        ax.set_title('(b) High Frequency(Bicubic)')

        if args.mag_plottype == '2d':
            ax = fig.add_subplot(3, 4, 3)
            ax.imshow(magnitude_spectrum_original, cmap='gray')
            ax.set_xlabel('Width')
            ax.set_ylabel('Height')
            ax.set_title('(c) Magnitude Spectrum(2D)')
        elif args.mag_plottype == '3d':
            ax = fig.add_subplot(3, 4, 3, projection='3d')
            original_spectrogram = np.meshgrid(np.arange(0, w_original), np.arange(0, h_original))
            ax.plot_surface(original_spectrogram[0], original_spectrogram[1], fft_aug_norm_original, cmap='gray')
            ax.set_xlabel('Width')
            ax.set_ylabel('Height')
            ax.set_zlabel('Frequency')
            ax.set_title('(c) Magnitude Spectrum(3D)')

        ax = fig.add_subplot(3, 4, 4)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        flat_fft_aug_norm_original = fft_aug_norm_original.copy().flatten()
        ax.hist(flat_fft_aug_norm_original[flat_fft_aug_norm_original>200])
        ax.set_xlabel('Pixel Values')
        ax.set_ylabel('Frequency')
        ax.set_title('(d) Pixel Values > 200')

        ax = fig.add_subplot(3, 4, 5)
        ax.imshow(cv2.cvtColor(roi_target, cv2.COLOR_BGR2RGB))
        ax.set_xlabel('Width')
        ax.set_ylabel('Height')
        ax.set_title('(a) SR ROI Image')

        ax = fig.add_subplot(3, 4, 6)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.imshow(HF_target, cmap='gray')
        ax.set_xlabel('Width')
        ax.set_ylabel('Height')
        ax.set_title('(b) High Frequency(SR)')

        if args.mag_plottype == '2d':
            ax = fig.add_subplot(3, 4, 7)
            ax.imshow(magnitude_spectrum_target, cmap='gray')
            ax.set_xlabel('Width')
            ax.set_ylabel('Height')
            ax.set_title('(c) Magnitude Spectrum(2D)')
        elif args.mag_plottype == '3d':
            ax = fig.add_subplot(3, 4, 7, projection='3d')
            target_spectrogram = np.meshgrid(np.arange(0, w_target), np.arange(0, h_target))
            ax.plot_surface(target_spectrogram[0], target_spectrogram[1], fft_aug_norm_target, cmap='gray')
            ax.set_xlabel('Width')
            ax.set_ylabel('Height')
            ax.set_zlabel('Frequency')
            ax.set_title('(c) Magnitude Spectrum')

        ax = fig.add_subplot(3, 4, 8)
        flat_fft_aug_norm_target = fft_aug_norm_target.copy().flatten()
        ax.hist(flat_fft_aug_norm_target[flat_fft_aug_norm_target > 200])
        ax.set_xlabel('Pixel Values')
        ax.set_ylabel('Frequency')
        ax.set_title('(d) Pixel Values > 200')

        ax = fig.add_subplot(3, 4, 9)
        ax.imshow(diff, cmap='gray')
        ax.set_title('(e) Diff')

        ax = fig.add_subplot(3, 4, 10)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        col_labels = ['Bicubic', 'SR']
        row_labels = ['BRISQUE', 'HF Pixel Count']
        table_vals = [[round(brisque.score(Image.fromarray(roi_original)), 2), round(brisque.score(Image.fromarray(roi_target)), 2)],
                      [len(flat_fft_aug_norm_original[flat_fft_aug_norm_original>200]), len(flat_fft_aug_norm_target[flat_fft_aug_norm_target > 200])]]

        # Draw table
        the_table = plt.table(cellText=table_vals,
                              colWidths=[0.1] * 3,
                              rowLabels=row_labels,
                              colLabels=col_labels,
                              loc='center')
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(9)
        the_table.scale(3, 2)

        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
        for pos in ['right', 'top', 'bottom', 'left']:
            plt.gca().spines[pos].set_visible(False)

        ax = fig.add_subplot(3, 4, 11)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.text(0.05, 0.2, '* Plot Info\n\n'
                           '  (b): High frequency pixels from (a).\n        (Shows sharp elements)\n'
                          '  (c): Sharp image if clearer points or lines\n        visible more often.\n'
                          '  (d): Number of sharp pixels\n        (Higher is better)\n'
                          '  (e): Difference between Bicubic/SR image', fontsize=9, color='black')

        fig.subplots_adjust(hspace=0.5, wspace=0.3)
        plt.show()

    return

def main(args):
    global original, original_copy, target, target_copy, original_BRISQUE, target_BRISQUE

    original = cv2.imread(args.original_path) # Source Image
    target = cv2.imread(args.target_path) # SR Image
    target_height, target_width, target_channel = target.shape

    original = cv2.resize(original, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
    original_copy = original.copy()
    target_copy = target.copy()

    cv2.namedWindow('Image Comparison')
    cv2.setMouseCallback('Image Comparison', onMouse, param=None)

    while True:
        cv2.resizeWindow('Image Comparison', imshow_width, imshow_height)
        cv2.imshow('Image Comparison', cv2.resize(original.copy(), (imshow_width, imshow_height)))

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--original_path', type=str, default='./test_imgs/bicubic.png') # Put Bicubic Upsampled Image
    parser.add_argument('--target_path', type=str, default='./test_imgs/bsrgan.png') # Put SR Image
    parser.add_argument('--mag_plottype', type=str, default='2d', choices=['2d', '3d']) # Plot type of magnitude spectrum
    args = parser.parse_args()

    main(args)
