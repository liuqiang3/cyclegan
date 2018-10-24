import tensorflow as tf
import cv2
tf.image.ssim_multiscale
image_1_raw_data = tf.gfile.FastGFile('/media/root/4c3b1a1e-e2bc-42fe-aeff-321d032e5c70/9.27/cyc_img/fake_b_0_0.jpg','rb').read()
image_2_raw_data = tf.gfile.FastGFile('/media/root/4c3b1a1e-e2bc-42fe-aeff-321d032e5c70/9.27/cyc_img/input_a_0_0.jpg','rb').read()
print(image_1_raw_data)
im1 = tf.image.decode_jpeg(
    image_1_raw_data)
im2 = tf.image.decode_jpeg(
    image_2_raw_data)
input_A = tf.placeholder(tf.float32, [1, 505, 374, 3], name="input_A")
input_B = tf.placeholder(tf.float32, [1, 505, 374, 3], name="input_B")
ssim = tf.image.ssim(input_A, input_B, max_val=255)
with tf.Session() as sess:
    im1 = tf.image.decode_jpeg(
        image_1_raw_data)
    im2 = tf.image.decode_jpeg(
        image_2_raw_data)
    print(im2.eval())
    cv2.imshow('img',im1.eval())
    cv2.waitKey(1000)
    # ssim = tf.image.ssim(im1, im2, max_val=255)
    ssIm=sess.run(ssim,feed_dict={input_A:im1,input_B:im2})
print(ssIm)