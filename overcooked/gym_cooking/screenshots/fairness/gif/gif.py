from PIL import Image
import glob
import os




for alloc in range(10):
    n_frames = [2,3,4,5,6,7]
    for i in n_frames:
        # Create the frames
        frames = []
        pathname = "../alloc_"+str(alloc+1)+"_frame_"+"*.png"
        imgs = sorted(glob.glob(pathname), key=os.path.getmtime)
        # print(imgs)
        for idx in imgs[:i]:
            new_frame = Image.open(idx)
            frames.append(new_frame)

        # Save into a GIF file that loops forever
        frames[0].save('alloc'+str(alloc+1)+'_'+str(i)+'.gif', format='GIF',
                       append_images=frames[1:],
                       save_all=True,
                       duration=300, loop=0)
