import os
import numpy as np

def read_and_print_psnr(file_path):
    """
    Loads a saved numpy array of PSNR values and prints the individual 
    scores alongside basic statistics.
    """
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        return

    # Load the PSNR array
    psnr_array = np.load(file_path)

    if len(psnr_array) == 0:
        print("The loaded PSNR array is empty.")
        return

    print("="*40)
    print(f" PSNR VALUES ({len(psnr_array)} samples)")
    print("="*40)

    # Print individual values
    for idx, psnr in enumerate(psnr_array):
        print(f"Triplet {idx:03d}: {psnr:.2f} dB")

    # Print global statistics
    print("-" * 40)
    print(f"Average PSNR: {np.mean(psnr_array):.2f} dB")
    print(f"Minimum PSNR: {np.min(psnr_array):.2f} dB")
    print(f"Maximum PSNR: {np.max(psnr_array):.2f} dB")
    print("="*40)

if __name__ == "__main__":
    # Adjust this path to point to your actual test evaluation output directory
    target_file = "/home/projects/sipl-prj10826/DeepOpticsHDR_PyTorch/model_to_test/images/test_evaluation_triplets/psnr_values.npy"
    
    read_and_print_psnr(target_file)