from PIL import Image
import glob
import os




for alloc in range(27):
    n_farmes = 10
    # Create the frames
    frames = []
    pathname = "../alloc_"+str(alloc+1)+"_frame_"+"*.png"
    imgs = sorted(glob.glob(pathname), key=os.path.getmtime)
    # print(imgs)
    for i in imgs[:n_farmes]:
        new_frame = Image.open(i)
        frames.append(new_frame)

    # Save into a GIF file that loops forever
    frames[0].save(''+str(alloc+1)+'.gif', format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=300, loop=0)
