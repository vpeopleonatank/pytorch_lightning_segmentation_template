import cv2
import torch
import numpy as np

sz = 128
N = 16


def get_tiles(img, mask):
    result = []
    shape = img.shape
    pad0, pad1 = (sz - shape[0] % sz) % sz, (sz - shape[1] % sz) % sz
    img = np.pad(
        img,
        [[pad0 // 2, pad0 - pad0 // 2], [pad1 // 2, pad1 - pad1 // 2], [0, 0]],
        constant_values=255,
    )
    mask = np.pad(
        mask,
        [[pad0 // 2, pad0 - pad0 // 2], [pad1 // 2, pad1 - pad1 // 2], [0, 0]],
        constant_values=0,
    )
    img = img.reshape(img.shape[0] // sz, sz, img.shape[1] // sz, sz, 3)
    img = img.transpose(0, 2, 1, 3, 4).reshape(-1, sz, sz, 3)
    mask = mask.reshape(mask.shape[0] // sz, sz, mask.shape[1] // sz, sz, 3)
    mask = mask.transpose(0, 2, 1, 3, 4).reshape(-1, sz, sz, 3)
    if len(img) < N:
        mask = np.pad(mask, [[0, N - len(img)], [0, 0], [0, 0], [0, 0]], constant_values=0)
        img = np.pad(img, [[0, N - len(img)], [0, 0], [0, 0], [0, 0]], constant_values=255)
    idxs = np.argsort(img.reshape(img.shape[0], -1).sum(-1))[:N]
    img = img[idxs]
    mask = mask[idxs].astype(np.uint8)
    for i in range(len(img)):
        result.append({"img": img[i], "mask": mask[i][:, :, 0].reshape(mask[i].shape[0], mask[i].shape[1]), "idx": i})
    return result


def show_images_by_tiles(img_path, label_path, n_tiles=16, image_size=224):

    big_img = cv2.imread(img_path)
    big_img = cv2.cvtColor(big_img, cv2.COLOR_BGR2RGB)
    big_mask = cv2.imread(label_path)
    # big_mask = cv2.cvtColor(big_mask, cv2.COLOR_BGR2RGB)
    # big_mask = big_mask[:, :, 0]
    # label 0: background, 1: gleason3, 2: gleason4, 3: gleason5
    # big_mask[(big_mask==1) | (big_mask==2)] = 0
    # big_mask[big_mask == 3] = 1
    # big_mask[big_mask == 4] = 2
    # big_mask[big_mask == 5] = 3

    # skimage.io.imsave("biopsy.png", big_img)
    # skimage.io.imsave("biopsy_label.png", big_mask)

    tiles = get_tiles(big_img, big_mask)

    idxes = list(range(n_tiles))

    n_row_tiles = int(np.sqrt(n_tiles))
    images = np.zeros((image_size * n_row_tiles, image_size * n_row_tiles, 3))
    masks = np.zeros((image_size * n_row_tiles, image_size * n_row_tiles), dtype=np.uint8)
    for h in range(n_row_tiles):
        for w in range(n_row_tiles):
            i = h * n_row_tiles + w

            if len(tiles) > idxes[i]:
                this_img = tiles[idxes[i]]["img"]
                this_mask = tiles[idxes[i]]["mask"]
            else:
                this_img = np.ones((image_size, image_size, 3)).astype(np.uint8) * 255
                this_mask = np.ones((image_size, image_size)).astype(np.uint8) * 0
            this_img = 255 - this_img
            h1 = h * image_size
            w1 = w * image_size
            images[h1 : h1 + image_size, w1 : w1 + image_size] = this_img
            masks[h1 : h1 + image_size, w1 : w1 + image_size] = this_mask
            # cv2.imshow("debug", this_img)
            # if cv2.waitKey(0) & 0xFF == ord("q"):
            #     continue
            # cv2.imwrite("debug.png", this_itrain_imagesmg)

    images = images.astype(np.float32)
    images /= 255

    cv2.imwrite("out.png", images)
    # out_label_name = f"{self.labels_out_path}{file_name_label}"
    # cv2.imwrite(out_label_name, masks)


def main():
    show_images_by_tiles(
        "/mnt/Data/data/PANDA_dataset/train_images/0005f7aaab2800f6170c399693a96917.png",
        "/mnt/Data/data/PANDA_dataset/train_label_masks/0005f7aaab2800f6170c399693a96917_mask.png",
        n_tiles=N,
        image_size=sz,
    )


if __name__ == "__main__":
    main()


# return datas, targets

# def __getitem__(self, idx):
#     file_name = self.df['image_id'].values[idx]
#     file_name_label = self.labels.values[idx]
#     file_path = f'{self.images_in_path}{file_name}.png'
#     file_path_label = f'{self.labels_in_path}{file_name_label}'
#     print(file_path)
#     # file_path = f'{TRAIN_IMAGES_PATH}006f6aa35a78965c92fffd1fbd53a058.png'
#     # file_path_label = f'{TRAIN_LABELS_PATH}006f6aa35a78965c92fffd1fbd53a058_mask.png'

#     big_img = cv2.imread(file_path)
#     big_img = cv2.cvtColor(big_img, cv2.COLOR_BGR2RGB)
#     big_img = big_img.transpose(2, 0, 1)
#     big_mask = cv2.imread(file_path_label)

#     return torch.tensor(big_img), torch.tensor(big_mask)
#     # return torch.tensor(images), torch.tensor(masks)
