import cv2
img = cv2.imread('/media/root/4c3b1a1e-e2bc-42fe-aeff-321d032e5c70/eye/messidor/1/11/20051019_38557_0100_PP.tif')
shape = img.shape
height = shape[1]
weight = shape[0]
crop = int((height-weight)/2)
img = img[0:weight,crop:(height-crop)]
