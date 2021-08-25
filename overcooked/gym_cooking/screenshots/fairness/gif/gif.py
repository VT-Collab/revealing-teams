from PIL import Image
import glob
import os
import pickle

def savedActions(gstar_idx):
    filename = "../../../data/fairness/actions_"+str(gstar_idx+1)+".pkl"
    actions = pickle.load(open(filename, "rb"))
    return actions


for gstar_idx in range(9):
    # Create the frames
    actions = savedActions(gstar_idx)
    lens = []
    for key in actions:
        lens.append(len(actions[key]))
    n_frames = [max(lens)]
    for i in n_frames:
        frames = []
        pathname = "../alloc_"+str(gstar_idx+1)+"_frame_"+"*.png"
        imgs = sorted(glob.glob(pathname), key=os.path.getmtime)
        # print(imgs)
        for idx in imgs[:i]:
            new_frame = Image.open(idx)
            frames.append(new_frame)

        # Save into a GIF file that loops forever
        frames[0].save('alloc'+str(gstar_idx+1)+'_'+str(i)+'.gif', format='GIF',
                       append_images=frames[1:],
                       save_all=True,
                       duration=300, loop=0)
