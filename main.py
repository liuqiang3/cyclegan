# from cyclegan_train_identity_loss import train
# from cyclegan_train import train
from cyclegan_train_ssim_v2 import train

from creat_tfrecord import creat_tfrecord
if __name__ == '__main__':



    #img.shape='512x512x3'
    #造tfrecord文件
    # creat_tfrecord(image_path='/media/root/4c3b1a1e-e2bc-42fe-aeff-321d032e5c70/try/img/b/*.jpg',
    #                out_put='/media/root/4c3b1a1e-e2bc-42fe-aeff-321d032e5c70/try/img/tf/traina.tfrecords')
    # creat_tfrecord(image_path='/media/root/4c3b1a1e-e2bc-42fe-aeff-321d032e5c70/try/img/a/*.jpg',
    #                out_put='/media/root/4c3b1a1e-e2bc-42fe-aeff-321d032e5c70/try/img/tf/trainb.tfrecords')


    train(
        filename_A='/media/root/4c3b1a1e-e2bc-42fe-aeff-321d032e5c70/9.27/tf/chase.tfrecords',
        filename_B='/media/root/4c3b1a1e-e2bc-42fe-aeff-321d032e5c70/9.27/tf/driver.tfrecords',
        record="/media/root/4c3b1a1e-e2bc-42fe-aeff-321d032e5c70/9.27/record",
        save_model="/media/root/4c3b1a1e-e2bc-42fe-aeff-321d032e5c70/9.27/model/model.ckpt",
        save_img="/media/root/4c3b1a1e-e2bc-42fe-aeff-321d032e5c70/9.27/cyc_img/",
        save_time =1,
        epochs =1000)
