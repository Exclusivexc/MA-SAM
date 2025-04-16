import pickle
import SimpleITK as sitk


path = "/media/ubuntu/maxiaochuan/MA-SAM/data/Promise12/Fore_slices/Case00_slice13.nii.gz"
img = sitk.GetArrayFromImage(sitk.ReadImage(path))
print(img.shape)

# with open(path, "rb") as file:
#     img = pickle.load(file)
#     print(img.shape)
