import torch.utils.data as data
import os
import lycon
from PIL import Image
import numpy as np
import torchvision

def correct_mask(mask):
    '''
    Replace 127 with 0 for cropping purposes
    '''
    mask_arr = np.array(mask)
    mask_arr[mask_arr==127]=0
    return Image.fromarray(mask_arr)

class TinyImageNetWithDepth(data.Dataset):
    def __init__(self, data_dir,depth_dir, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])):
        
        def get_imgs_paths(dirr):
            with open('id_test.txt','r') as f:
                class_paths = [os.path.join(dirr,clas) for clas in  list(f.readlines())]
                img_paths = []
                for class_path in class_paths :
                    img_paths = img_paths + [os.path.join(class_path, img) for img in os.listdir(class_path)]        
                return img_paths

        super(TinyImageNetWithDepth, self).__init__()
        self.data_dir = os.path.expanduser(data_dir)
        self.img_paths = get_imgs_paths(self.data_dir)
        self.depth_paths = [path.replace(data_dir,depth_dir).replace('JPEG','png') for path in self.img_paths]
        self.transform = transform

    def __getitem__(self, index):

        imgpath = self.img_paths[index]
        depthpath = self.depth_paths[index]
        img = Image.fromarray(lycon.load(imgpath)).resize((244,244))
        depth = Image.fromarray(lycon.load(depthpath)[:,:,0]).resize((244,244))
        

        
        if self.transform is not None:
            img = self.transform(img)
            depth = self.transform(depth)
        return img,depth

    def __len__(self):
        return len(self.img_paths)

if __name__=='__main__':
    import matplotlib.pyplot as plt

    dataset = TinyImageNetWithDepth('./images','./depthmaps')

    idx = 8

    img, depth = dataset.__getitem__(idx)

    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(img.permute(1,2,0))
    plt.subplot(1,2,2)
    plt.imshow(depth.squeeze())

    plt.show()
    print(depth)