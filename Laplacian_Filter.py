import sys
import cv2 as cv
import matplotlib.pyplot as plt


def init(argv):
    # [variables]
    # Declare the variables we are going to use
    ddepth = cv.CV_16S
    kernel_size = 3
    window_name = "Laplace Demo"
    # [variables]

    # [load]
    imageName = argv[0] if len(argv) > 0 else 'logit.jpg'

    src = cv.imread(cv.samples.findFile(imageName),
                    cv.IMREAD_COLOR)  # Load an image

    # Check if image is loaded fine
    if src is None:
        print('Error opening image')
        print('Program Arguments: [image_name -- default logit.jpg]')
        return -1
    # [load]

    # [reduce_noise]
    # Remove noise by blurring with a Gaussian filter
    src = cv.GaussianBlur(src, (3, 3), 0)
    # [reduce_noise]

    # [convert_to_gray]
    # Convert the image to grayscale
    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    # [convert_to_gray]

    # Create Window
    # cv.namedWindow(window_name, cv.WINDOW_AUTOSIZE)

    # [laplacian]
    # Apply Laplace function
    dst = cv.Laplacian(src_gray, ddepth, ksize=kernel_size)
    # [laplacian]

    # [convert]
    # converting back to uint8
    abs_dst = cv.convertScaleAbs(dst)
    # [convert]

    # [display]
    # cv.imshow(window_name, abs_dst)
    # cv.waitKey(0)
    # [display]

    return abs_dst


def hien_thi_LoG():
    # định nghĩa vùng vẽ ảnh với phương thức figure
    fig = plt.figure(figsize=(16, 9))
    # tạo hai vùng vẽ ảnh
    ax1, ax2 = fig.subplots(1, 2)

    # su dung phuong thuc imread để mở ảnh
    img = cv.imread('logit.jpg', 0)
    # hien thi anh trong vùng ax1
    ax1.imshow(img, cmap='gray')
    ax1.set_title("ẢNH GỐC")

    after = init(sys.argv[1:])
    ax2.imshow(after, cmap='gray')
    ax2.set_title("ẢNH SAU KHI LỌC LoG")
    plt.show()


if __name__ == "__main__":
    hien_thi_LoG()
