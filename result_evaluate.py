import os
import glob
import numpy as np
import pydicom
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# =========================================================
# [설정] 경로 설정
# =========================================================

# 1. 생성된(Fake) DICOM 파일 경로
FAKE_B_ROOT = "/mnt/nas100/forGPU/bc_cho/2_Code/ResViT/results/ct_contrast_pretrain/test_latest/dcm"

# 2. 정답(Real) DICOM 파일 경로
REAL_B_ROOT = "/mnt/nas206/ANO_DET/GAN_body/Pulmonary_Embolism/sampled_data/CCY_PE_DECT/journal_data/internal/test/70keV"

# 3. 결과 엑셀 파일 저장 이름
OUTPUT_EXCEL_NAME = "evaluation_results_v2.xlsx"

# =========================================================

def read_dicom_to_hu(path):
    """DICOM 파일을 읽어 HU(Hounsfield Unit) 값의 Numpy 배열로 변환"""
    try:
        dcm = pydicom.dcmread(path, force=True)
        img = dcm.pixel_array.astype(np.float32)
        
        slope = getattr(dcm, 'RescaleSlope', 1)
        intercept = getattr(dcm, 'RescaleIntercept', 0)
        img = img * slope + intercept
        return img
    except Exception as e:
        return None

def evaluate():
    print(f"Searching for DICOM files in: {FAKE_B_ROOT}")
    
    # 1. 파일 탐색
    fake_files = glob.glob(os.path.join(FAKE_B_ROOT, "**", "*.dcm"), recursive=True)
    fake_files += glob.glob(os.path.join(FAKE_B_ROOT, "**", "*.DCM"), recursive=True)
    fake_files = list(set(fake_files))
    
    # fake_B 폴더 내 파일만 필터링
    fake_files = [f for f in fake_files if "fake_B" in f]

    if not fake_files:
        print("❌ DICOM 파일을 찾을 수 없습니다.")
        return

    print(f"Found {len(fake_files)} files. Starting evaluation...")
    
    results_data = []
    count = 0
    missing_real_files = 0

    for fake_path in fake_files:
        filename = os.path.basename(fake_path)            # PE275_0001.dcm
        filename_no_ext = os.path.splitext(filename)[0]   # PE275_0001
        patient_id = filename_no_ext.split('_')[0]        # PE275
        
        # 소스 에너지 레벨 추출 (경로 구조: .../80keV/PE275/fake_B/...)
        # fake_path의 상위(fake_B) -> 상위(PE275) -> 상위(80keV) 폴더명 추출
        try:
            source_kev = os.path.basename(os.path.dirname(os.path.dirname(fake_path))) # 예: 80keV
            # 만약 경로 구조가 다르면 'Unknown'으로 처리
            if "keV" not in source_kev: 
                source_kev = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(fake_path))))
        except:
            source_kev = "Unknown"

        # 정답 파일 경로 매칭
        real_filename = f"{filename_no_ext}_70 keV.dcm"
        real_path = os.path.join(REAL_B_ROOT, patient_id, real_filename)
        
        if not os.path.exists(real_path):
            alt_real_filename = f"{filename_no_ext}_70keV.dcm"
            alt_path = os.path.join(REAL_B_ROOT, patient_id, alt_real_filename)
            
            if os.path.exists(alt_path):
                real_path = alt_path
            else:
                missing_real_files += 1
                if missing_real_files <= 5:
                    print(f"⚠️ GT File missing: {real_path}")
                continue

        # 이미지 로드
        fake_img = read_dicom_to_hu(fake_path)
        real_img = read_dicom_to_hu(real_path)

        if fake_img is None or real_img is None:
            continue

        if fake_img.shape != real_img.shape:
            continue

        # Metric 계산
        data_range = 4095.0 # HU Range
        
        # 1. PSNR
        val_psnr = psnr(real_img, fake_img, data_range=data_range)
        # 2. SSIM
        val_ssim = ssim(real_img, fake_img, data_range=data_range)
        # 3. MAE (Mean Absolute Error) 추가
        val_mae = np.mean(np.abs(real_img - fake_img))

        # 데이터 추가
        results_data.append({
            "Source_keV": source_kev,
            "Patient_ID": patient_id,
            "Filename": filename,
            "PSNR": val_psnr,
            "SSIM": val_ssim,
            "MAE": val_mae,  # 엑셀 컬럼 추가
            "Real_Path": real_path
        })
        
        count += 1
        if count % 100 == 0:
            print(f"Processed {count} files... (Avg PSNR: {np.mean([d['PSNR'] for d in results_data]):.2f}, Avg MAE: {np.mean([d['MAE'] for d in results_data]):.2f})")

    # =========================================================
    # [엑셀 저장]
    # =========================================================
    if len(results_data) > 0:
        print("\nCreating Excel file...")
        
        df = pd.DataFrame(results_data)
        
        # 컬럼 순서 재배치 (보기 좋게)
        cols = ["Source_keV", "Patient_ID", "Filename", "PSNR", "SSIM", "MAE", "Real_Path"]
        df = df[cols]

        # 전체 평균 계산
        avg_psnr = df['PSNR'].mean()
        avg_ssim = df['SSIM'].mean()
        avg_mae = df['MAE'].mean()
        
        print("-" * 30)
        print(f"Total: {len(df)}")
        print(f"Avg PSNR: {avg_psnr:.4f}")
        print(f"Avg SSIM: {avg_ssim:.4f}")
        print(f"Avg MAE : {avg_mae:.4f}")
        print("-" * 30)

        # 평균 행 추가
        avg_row = pd.DataFrame([{
            "Source_keV": "AVERAGE",
            "Patient_ID": "-",
            "Filename": "-",
            "PSNR": avg_psnr,
            "SSIM": avg_ssim,
            "MAE": avg_mae,
            "Real_Path": "-"
        }])
        
        df = pd.concat([df, avg_row], ignore_index=True)
        
        # 엑셀 파일 저장
        df.to_excel(OUTPUT_EXCEL_NAME, index=False)
        print(f"\n✅ Excel saved successfully: {os.path.abspath(OUTPUT_EXCEL_NAME)}")
        
    else:
        print("\n❌ 저장할 데이터가 없습니다.")

if __name__ == "__main__":
    evaluate()