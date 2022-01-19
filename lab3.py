import cv2
import numpy as np
'''
 Hướng dẫn các tham số khi run chương trình
 -d0: filter size D0
 -flag: filter type
 0-ideal filtering (Lọc lý tưởng)
 1-Butterworth filtering (lọc Butter)
 2-Gaussian filtering (lọc Gauss)
 -n: order of Butterworth filtering 
 -lh: low-pass filtering or high-pass filtering (Theo thông thấp hoặc cao)
 Filtered Image window: filtered image and filter image
'''
 
 
def combine_images(images, axis=1):
    '''
         Combine images.
         @param images: image list (image members must have the same dimension)
         @param axis: merge direction. 
         When axis = 0, the images are merged vertically;
         When axis = 1, the images are merged horizontally.
         @return merged image
    '''
    ndim = images[0].ndim
    shapes = np.array([mat.shape for mat in images])
    assert np.all(map(lambda e: len(e) == ndim, shapes)
                  ), 'all images should be same ndim.'
         if axis == 0: # merge images vertically
                 # Merge image cols
        cols = np.max(shapes[:, 1])
                 # Expand the cols size of each image to make the cols consistent
        copy_imgs = [cv2.copyMakeBorder(img, 0, 0, 0, cols - img.shape[1],
                                        cv2.BORDER_CONSTANT, (0, 0, 0)) for img in images]
                 # Merge vertically
        return np.vstack(copy_imgs)
         else: # merge images horizontally
                 # Combine the rows of the image
        rows = np.max(shapes[:, 0])
                 # Expand the row size of each image to make rows consistent
        copy_imgs = [cv2.copyMakeBorder(img, 0, rows - img.shape[0], 0, 0,
                                        cv2.BORDER_CONSTANT, (0, 0, 0)) for img in images]
                 # Merge horizontally
        return np.hstack(copy_imgs)
 
 
def fft(img):
         '' 'Fourier transform the image and return the frequency matrix after transposition' ''
    assert img.ndim == 2, 'img should be gray.'
    rows, cols = img.shape[:2]
         # Calculate the optimal size
    nrows = cv2.getOptimalDFTSize(rows)
    ncols = cv2.getOptimalDFTSize(cols)
         # According to the new size, create a new transformed image
    nimg = np.zeros((nrows, ncols))
    nimg[:rows, :cols] = img
         # Fourier transform
    fft_mat = cv2.dft(np.float32(nimg), flags=cv2.DFT_COMPLEX_OUTPUT)
         # Transposition, the low frequency part moves to the middle, the high frequency part moves to the surrounding
    return np.fft.fftshift(fft_mat)
 
 
def fft_image(fft_mat):
         '' 'Convert frequency matrix to visual image' ''
         # Add 1 to the log function to avoid log (0).
    log_mat = cv2.log(1 + cv2.magnitude(fft_mat[:, :, 0], fft_mat[:, :, 1]))
         # Standardized to between 0 ~ 255
    cv2.normalize(log_mat, log_mat, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(np.around(log_mat))
 
 
def ifft(fft_mat):
         '' 'Inverse Fourier transform, return inverse transform image' ''
         # Reverse transposition, the low frequency part moves to the surrounding, the high frequency part moves to the middle
    f_ishift_mat = np.fft.ifftshift(fft_mat)
         # Inverse Fourier Transform
    img_back = cv2.idft(f_ishift_mat)
         # Convert complex number to amplitude, sqrt (re ^ 2 + im ^ 2)
    img_back = cv2.magnitude(*cv2.split(img_back))
         # Standardized to between 0 ~ 255
    cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(np.around(img_back))
 
 
def fft_distances(m, n):
    '''
         Calculate the distance of each point of the m, n matrix from the center
         See page 93 of "Digital Image Processing MATLAB Edition. Gonzalez"
    '''
    u = np.array([i if i <= m / 2 else m - i for i in range(m)],
                 dtype=np.float32)
    v = np.array([i if i <= m / 2 else m - i for i in range(m)],
                 dtype=np.float32)
    v.shape = n, 1
         # The distance from each point to the upper left corner of the matrix
    ret = np.sqrt(u * u + v * v)
         # The distance of each point from the center of the matrix
    return np.fft.fftshift(ret)
 
 
def lpfilter(flag, rows, cols, d0, n):
         '''Low-pass filter
         @param flag: filter type
         0-ideal low-pass filtering
         1-Butterworth low-pass filtering
         2-Gaussian low-pass filtering
         @param rows: the height of the filtered matrix
         @param cols: the width of the filtered matrix
         @param d0: filter size D0
         @param n: order of Butterworth low-pass filtering 
         @return filter matrix 

    '''
    assert d0 > 0, 'd0 should be more than 0.'
    filter_mat = None
         # Ideal low-pass filtering
    if flag == 0:
        filter_mat = np.zeros((rows, cols, 2), np.float32)
        cv2.circle(filter_mat, (rows / 2, cols / 2),
                   d0, (1, 1, 1), thickness=-1)
         # Butterworth low-pass filtering
    elif flag == 1:
        duv = fft_distances(*fft_mat.shape[:2])
        filter_mat = 1 / (1 + np.power(duv / d0, 2 * n))
                 # fft_mat has 2 channels, real and imaginary
                 # fliter_mat also requires 2 channels
        filter_mat = cv2.merge((filter_mat, filter_mat))
         # Gaussian low-pass filtering
    else:
        duv = fft_distances(*fft_mat.shape[:2])
        filter_mat = np.exp(-(duv * duv) / (2 * d0 * d0))
                 # fft_mat has 2 channels, real and imaginary
                 # fliter_mat also requires 2 channels
        filter_mat = cv2.merge((filter_mat, filter_mat))
    return filter_mat
 
 
def hpfilter(flag, rows, cols, d0, n):
         '''High-pass filter
         @param flag: filter type
         0-ideal high-pass filtering
         1-Butterworth high-pass filtering
         2-Gaussian high-pass filtering
         @param rows: the height of the filtered matrix
         @param cols: the width of the filtered matrix
         @param d0: filter size D0
         @param n: the order of Butterworth high-pass filtering 
         @return filter matrix 
    '''
    assert d0 > 0, 'd0 should be more than 0.'
    filter_mat = None
         # Ideal high-pass filtering
    if flag == 0:
        filter_mat = np.ones((rows, cols, 2), np.float32)
        cv2.circle(filter_mat, (rows / 2, cols / 2),
                   d0, (0, 0, 0), thickness=-1)
         # Butterworth high-pass filtering
    elif flag == 1:
        duv = fft_distances(rows, cols)
         # duv has a value of 0 (the center is 0 from the center). To avoid division by 0, set the center to 0.000001
        duv[rows / 2, cols / 2] = 0.000001
        filter_mat = 1 / (1 + np.power(d0 / duv, 2 * n))
                 # fft_mat has 2 channels, real and imaginary
                 # fliter_mat also requires 2 channels
        filter_mat = cv2.merge((filter_mat, filter_mat))
         # Gaussian high-pass filtering
    else:
        duv = fft_distances(*fft_mat.shape[:2])
        filter_mat = 1 - np.exp(-(duv * duv) / (2 * d0 * d0))
                 # fft_mat has 2 channels, real and imaginary
                 # fliter_mat also requires 2 channels
        filter_mat = cv2.merge((filter_mat, filter_mat))
    return filter_mat
 
 
def do_filter(_=None):
         '' 'Filter and display' ''
    d0 = cv2.getTrackbarPos('d0', filter_win)
    flag = cv2.getTrackbarPos('flag', filter_win)
    n = cv2.getTrackbarPos('n', filter_win)
    lh = cv2.getTrackbarPos('lh', filter_win)
         # Filter
    filter_mat = None
    if lh == 0:
        filter_mat = lpfilter(flag, fft_mat.shape[0], fft_mat.shape[1], d0, n)
    else:
        filter_mat = hpfilter(flag, fft_mat.shape[0], fft_mat.shape[1], d0, n)
         # Perform filtering
    filtered_mat = filter_mat * fft_mat
         # Inverse transform
    img_back = ifft(filtered_mat)
         # Display filtered image and filter image
    cv2.imshow(image_win, combine_images([img_back, fft_image(filter_mat)]))
if __name__ == '__main__':
    img = cv2.imread('hoa.jpg', 0)
    rows, cols = img.shape[:2]
         # Filter window name
    filter_win = 'Filter Parameters'
         # Image window name
    image_win = 'Filtered Image'
    cv2.namedWindow(filter_win)
    cv2.namedWindow(image_win)
         # Create d0 tracker, d0 is the filter size
    cv2.createTrackbar('d0', filter_win, 20, min(rows, cols) / 4, do_filter)
         # Create flag tracker,
    # flag = 0, ideal filtering
         # flag = 1, filter for Butterworth
         # flag = 2, Gaussian filtering
    cv2.createTrackbar('flag', filter_win, 0, 2, do_filter)
         # Create n tracker
         # n is the order of Butterworth filtering
    cv2.createTrackbar('n', filter_win, 1, 5, do_filter)
         # Create lh tracker
         # lh: whether the filter is low pass or high pass, 0 is low pass, 1 is high pass
    cv2.createTrackbar('lh', filter_win, 0, 1, do_filter)
    fft_mat = fft(img)
    do_filter()
    cv2.resizeWindow(filter_win, 512, 20)
    cv2.waitKey(0)
    cv2.destroyAllWindows()