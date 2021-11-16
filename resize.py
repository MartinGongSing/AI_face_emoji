import cv2

emoji = cv2.imread("drop.jpg") #emoji


(w,h) = (250, 300)
dim = (w,h)
resized = cv2.resize(emoji, dim)
print(resized.shape)