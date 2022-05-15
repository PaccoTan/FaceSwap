import os
import glob
for i in range(15):
    # Get a list of all the file paths that ends with .txt from in specified directory
    fileList = glob.glob(f'./dataset/CelebAMask-HQ-mask-anno/skin/*skin*.png')
    # Iterate over the list of filepaths & remove each file.
    for filePath in fileList:
        try:
            #os.rename(filePath)
            temp = filePath.split("\\")[1].split('_')[0]
            print(f"./dataset/CelebAMask-HQ-mask-anno/skin/{temp}_hair_0.png")
            os.rename(filePath, f"./dataset/CelebAMask-HQ-mask-anno/skin/{temp}_skin.png")
           
        except:
            print("Error while deleting file : ", filePath)