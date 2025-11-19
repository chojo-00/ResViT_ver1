import os
import numpy as np
import nibabel as nib
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
import re


def parse_filename(filepath):
    """
    파일 경로에서 환자 ID와 슬라이스 번호 추출
    
    입력 예시:
    - /train/70keV/PE001/PE001_0001.dcm
    - /test/80keV/PE275/PE275_0280.dcm
    
    출력: ('PE001', '0001', '70keV')
    """
    # 파일명 추출
    basename = os.path.basename(filepath)  # PE001_0001.dcm
    
    # 확장자 제거
    name_without_ext = os.path.splitext(basename)[0]  # PE001_0001
    
    # 패턴: PE{환자번호}_{슬라이스번호}
    pattern = r'(PE\d+)_(\d+)'
    match = re.match(pattern, name_without_ext)
    
    if match:
        patient_id = match.group(1)    # PE001
        slice_num = match.group(2)     # 0001
    else:
        # 폴백: 파일명에서 숫자 추출 시도
        patient_id = "Unknown"
        slice_match = re.search(r'(\d+)', name_without_ext)
        slice_num = slice_match.group(1) if slice_match else '0000'
    
    # 경로에서 keV 정보 추출
    source_kev = 'unknownkeV'
    path_parts = filepath.split(os.sep)
    for part in path_parts:
        if 'keV' in part:
            # "70keV", "80keV" 등의 형식
            kev_match = re.search(r'(\d+)\s*keV', part, re.IGNORECASE)
            if kev_match:
                source_kev = f"{kev_match.group(1)}keV"
            else:
                source_kev = part
            break
    
    return patient_id, slice_num, source_kev


def tensor2array(image_tensor, min_hu=-1024.0, max_hu=3071.0):
    """
    Tensor를 numpy array로 변환하고 원본 CT HU 값으로 복원
    
    정규화 복원 과정:
    1. 모델 출력: [-1, 1] (Tanh 출력)
    2. [0, 1]로 변환: (tensor + 1) / 2
    3. 원본 HU 범위로 복원: normalized * (max_hu - min_hu) + min_hu
    
    Args:
        image_tensor: torch tensor [C, H, W] with values in [-1, 1]
        min_hu: 최소 HU 값 (전처리 시 사용한 값과 동일해야 함)
        max_hu: 최대 HU 값 (전처리 시 사용한 값과 동일해야 함)
    
    Returns:
        numpy array [H, W] with original CT HU values
    """
    # Tensor → Numpy
    image_numpy = image_tensor[0].cpu().float().numpy()
    
    if image_numpy.shape[0] == 1:
        # Single channel
        image_numpy = image_numpy[0]  # [H, W]
    else:
        # Multi-channel인 경우 첫 번째 채널만 사용
        image_numpy = image_numpy[0]
    
    # Step 1: [-1, 1] → [0, 1]
    image_numpy = (image_numpy + 1.0) / 2.0
    
    # Step 2: [0, 1] → [min_hu, max_hu]
    image_numpy = image_numpy * (max_hu - min_hu) + min_hu
    
    return image_numpy


def save_ct_image_both_formats(image_array, npy_dir, nii_dir, filename_base):
    """
    CT 이미지를 numpy와 nifti 형식으로 저장
    
    Args:
        image_array: numpy array [H, W] with HU values
        npy_dir: numpy 저장 디렉토리
        nii_dir: nifti 저장 디렉토리
        filename_base: 파일명 (확장자 제외)
    
    Returns:
        tuple: (npy_path, nii_path)
    """
    # 디렉토리 생성
    os.makedirs(npy_dir, exist_ok=True)
    os.makedirs(nii_dir, exist_ok=True)
    
    # 1. Numpy 저장 (.npy)
    npy_path = os.path.join(npy_dir, f"{filename_base}.npy")
    np.save(npy_path, image_array)
    
    # 2. NIfTI 저장 (.nii.gz)
    # 2D 이미지를 3D volume로 변환 [H, W] -> [H, W, 1]
    nifti_array = np.expand_dims(image_array, axis=-1)
    
    # NIfTI 이미지 생성 (affine은 단위 행렬)
    nifti_img = nib.Nifti1Image(nifti_array, affine=np.eye(4))
    
    # 저장
    nii_path = os.path.join(nii_dir, f"{filename_base}.nii.gz")
    nib.save(nifti_img, nii_path)
    
    return npy_path, nii_path


if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.nThreads = 1
    opt.batchSize = 1
    opt.serial_batches = True
    opt.no_flip = True

    # Source keV 리스트
    src_list = opt.src.split(',')
    
    # CT HU 값 범위 설정
    # 주의: 전처리 시 사용한 값과 동일해야 함!
    MIN_HU = -1024.0
    MAX_HU = 3071.0
    
    print(f"{'='*80}")
    print(f"⚙️  CT Value Range Settings")
    print(f"{'='*80}")
    print(f"Min HU: {MIN_HU}")
    print(f"Max HU: {MAX_HU}")
    print(f"Range: {MAX_HU - MIN_HU}")
    print(f"\n💡 These values should match the preprocessing settings!")
    print(f"   Check data/dect_dataset.py _CT_preprocess function")
    print(f"{'='*80}\n")
    
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
    
    # 결과 디렉토리 (base)
    results_base = os.path.join(opt.results_dir, opt.name, 
                                f'{opt.phase}_{opt.which_epoch}')
    
    # npy와 nii 각각의 base 디렉토리
    npy_base = os.path.join(results_base, 'npy')
    nii_base = os.path.join(results_base, 'nii')
    
    print(f"{'='*80}")
    print(f"🧪 Testing {opt.name}")
    print(f"{'='*80}")
    print(f"Source keV: {src_list}")
    print(f"Target keV: {opt.trg}")
    print(f"Total samples to test: {min(opt.how_many, len(dataset))}")
    print(f"Results directory: {results_base}")
    print(f"  - Numpy:  {npy_base}")
    print(f"  - NIfTI:  {nii_base}")
    print(f"{'='*80}\n")
    
    # 통계
    stats = {kev: {'patients': set(), 'slices': 0} for kev in src_list}
    
    # Test loop
    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break
        
        model.set_input(data)
        model.test()
        
        # 이미지 경로 가져오기
        img_paths = model.get_image_paths()
        img_path = img_paths[0] if isinstance(img_paths, list) else img_paths
        
        # 파일명에서 정보 추출
        patient_id, slice_num, source_kev = parse_filename(img_path)
        
        # 진행상황 출력
        if (i + 1) % 10 == 0 or i == 0:
            print(f'[{i+1:04d}/{min(opt.how_many, len(dataset))}] '
                  f'{source_kev} → 70keV | {patient_id} | slice {slice_num}')
        
        # 통계 업데이트
        if source_kev in stats:
            stats[source_kev]['patients'].add(patient_id)
            stats[source_kev]['slices'] += 1
        
        # 파일명: PE{환자번호}_{슬라이스번호}
        filename_base = f"{patient_id}_{slice_num}"
        
        # 이미지 가져오기 및 변환 (HU 값으로 복원)
        real_A = tensor2array(model.real_A.data, MIN_HU, MAX_HU)
        real_B = tensor2array(model.real_B.data, MIN_HU, MAX_HU)
        fake_B = tensor2array(model.fake_B.data, MIN_HU, MAX_HU)
        
        # 디렉토리 구조 생성
        # npy 경로
        npy_kev_dir = os.path.join(npy_base, source_kev)
        npy_patient_dir = os.path.join(npy_kev_dir, patient_id)
        npy_real_A_dir = os.path.join(npy_patient_dir, 'real_A')
        npy_real_B_dir = os.path.join(npy_patient_dir, 'real_B')
        npy_fake_B_dir = os.path.join(npy_patient_dir, 'fake_B')
        
        # nii 경로
        nii_kev_dir = os.path.join(nii_base, source_kev)
        nii_patient_dir = os.path.join(nii_kev_dir, patient_id)
        nii_real_A_dir = os.path.join(nii_patient_dir, 'real_A')
        nii_real_B_dir = os.path.join(nii_patient_dir, 'real_B')
        nii_fake_B_dir = os.path.join(nii_patient_dir, 'fake_B')
        
        # 저장 (npy + nii)
        save_ct_image_both_formats(real_A, npy_real_A_dir, nii_real_A_dir, filename_base)
        save_ct_image_both_formats(real_B, npy_real_B_dir, nii_real_B_dir, filename_base)
        save_ct_image_both_formats(fake_B, npy_fake_B_dir, nii_fake_B_dir, filename_base)
    
    # 최종 통계
    print(f"\n{'='*80}")
    print(f"✅ Testing Complete!")
    print(f"{'='*80}")
    print(f"\n📊 Statistics by Source keV:")
    print(f"{'-'*80}")
    print(f"{'keV':<12} {'Patients':<15} {'Slices':<10}")
    print(f"{'-'*80}")
    
    total_slices = 0
    all_patients = set()
    
    for kev in src_list:
        if kev in stats:
            num_patients = len(stats[kev]['patients'])
            num_slices = stats[kev]['slices']
            total_slices += num_slices
            all_patients.update(stats[kev]['patients'])
            print(f"{kev:<12} {num_patients:<15} {num_slices:<10}")
    
    print(f"{'-'*80}")
    print(f"{'Total':<12} {len(all_patients):<15} {total_slices:<10}")
    print(f"\n📁 Results saved to: {results_base}")
    print(f"   ├── npy/  (numpy arrays)")
    print(f"   └── nii/  (NIfTI format)")
    print(f"\n💾 File formats:")
    print(f"   - .npy:    Numpy arrays with original HU values [{MIN_HU}, {MAX_HU}]")
    print(f"   - .nii.gz: NIfTI format for medical imaging software (ITK-SNAP, 3D Slicer)")
    print(f"\n💡 Example paths:")
    print(f"   {npy_base}/80keV/PE001/real_A/PE001_0001.npy")
    print(f"   {nii_base}/80keV/PE001/fake_B/PE001_0001.nii.gz")
    print(f"\n💡 To load:")
    print(f"   # Numpy")
    print(f"   img = np.load('PE001_0001.npy')  # Shape: [H, W], dtype: float32")
    print(f"   # NIfTI")
    print(f"   nii = nib.load('PE001_0001.nii.gz')")
    print(f"   img = nii.get_fdata()  # Shape: [H, W, 1]")