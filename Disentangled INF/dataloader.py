import os
import pytorch_lightning as ptl
from torch.utils.data import DataLoader, Dataset
import nibabel as nib
import numpy as np


class UCSF_PGDM_Dataset(Dataset):
    def __init__(self):
        self.data_dir = "../datasets/Subsample_UCSF-PDGM_Dataset/"
        self.dataset = sorted(os.listdir(self.data_dir))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        patient_dir = self.dataset[idx]
        patient_id = patient_dir[:-6]
        patient_data = sorted(os.listdir(self.data_dir + patient_dir))
        brain_mask = list(filter(lambda x: 'brain_segmentation.nii.gz' in x, patient_data))[0]
        brain_mask = nib.load(self.data_dir + patient_dir + '/' + brain_mask)
        tumor_seg = list(filter(lambda x: 'tumor_segmentation.nii.gz' in x, patient_data))[0]
        tumor_seg = nib.load(self.data_dir + patient_dir + '/' + tumor_seg)
        t1 = list(filter(lambda x: 'T1.nii.gz' in x, patient_data))[0]
        t1 = nib.load(self.data_dir + patient_dir + '/' + t1)
        t1c = list(filter(lambda x: 'T1c.nii.gz' in x, patient_data))[0]
        t1c = nib.load(self.data_dir + patient_dir + '/' + t1c)
        t2 = list(filter(lambda x: 'T2.nii.gz' in x, patient_data))[0]
        t2 = nib.load(self.data_dir + patient_dir + '/' + t2)
        flair = list(filter(lambda x: 'FLAIR.nii.gz' in x, patient_data))[0]
        flair = nib.load(self.data_dir + patient_dir + '/' + flair)
        swi = list(filter(lambda x: 'SWI.nii.gz' in x, patient_data))[0]
        swi = nib.load(self.data_dir + patient_dir + '/' + swi)
        dwi = list(filter(lambda x: 'DWI.nii.gz' in x, patient_data))[0]
        dwi = nib.load(self.data_dir + patient_dir + '/' + dwi)

        return (patient_id,
                brain_mask.get_fdata(),
                tumor_seg.get_fdata(),
                t1.get_fdata(),
                t1c.get_fdata(),
                t2.get_fdata(),
                flair.get_fdata(),
                swi.get_fdata(),
                dwi.get_fdata())


class UCSF_PGDM_DataModule(ptl.LightningDataModule):
    def __init__(self, batchSize, n_workers):
        super(UCSF_PGDM_DataModule, self).__init__()
        self.save_hyperparameters()
        self.batchSize = batchSize
        self.n_workers = n_workers

    def train_dataloader(self):
        self.train_dataset = UCSF_PGDM_Dataset()
        return DataLoader(self.train_dataset,
                          batch_size=self.batchSize,
                          shuffle=False,
                          num_workers=self.n_workers,
                          pin_memory=True)


def main():
    # Main function provides unit testing for the dataloader
    dataset = UCSF_PGDM_Dataset()

    print(f'Testing Dataset: {dataset.__class__.__name__}')
    print(f'Dataset Length: {len(dataset)}')
    for batch_idx, data in enumerate(dataset):
        patient_id, brain_mask, tumor_seg, t1, t1c, t2, flair, swi, dwi = data
        print(f'Patient ID: {patient_id}, dtype: {type(patient_id)}')
        print(f'Brain Mask Shape: {brain_mask.shape}, dtype: {brain_mask.dtype}')
        print(f'Tumor Segmentation Shape: {tumor_seg.shape}')
        print(f'T1 Shape: {t1.shape}')
        print(f'T1c Shape: {t1c.shape}')
        print(f'T2 Shape: {t2.shape}')
        print(f'FLAIR Shape: {flair.shape}')
        print(f'SWI Shape: {swi.shape}')
        print(f'DWI Shape: {dwi.shape}')
        print('\n')
        if batch_idx == 10:
            break

    dm = UCSF_PGDM_DataModule(batchSize=2, n_workers=4)
    print(f'Testing DataModule: {UCSF_PGDM_DataModule.__name__}')



if __name__ == '__main__':
    main()
