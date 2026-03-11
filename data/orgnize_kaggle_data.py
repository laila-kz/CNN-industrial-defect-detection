import pandas as pd
import os
import shutil
from pathlib import Path
import numpy as np
import yaml

def organize_kaggle_steel_data():
    """
    Organize Kaggle steel defect dataset into train/val/test OK/DEFECT folders
    Based on your file structure:
    data/
      test_images/
      train_images/
      train.csv
    """
    
    # ============================================
    # 1. DEFINE PATHS (UPDATE THESE IF NEEDED)
    # ============================================
    
    # Get current directory where script is running
    current_dir = os.getcwd()
    print(f"📂 Current directory: {current_dir}")
    
    # Kaggle dataset is in 'data' folder in current directory
    kaggle_path = os.path.join(current_dir, "data")
    
    # Output will be in 'data/organized' folder
    output_path = os.path.join(current_dir, "data", "organized")
    
    print(f"📁 Kaggle input path: {kaggle_path}")
    print(f"📁 Output path: {output_path}")
    
    # ============================================
    # 2. DEFINE SPECIFIC FILE PATHS
    # ============================================
    
    train_images_dir = os.path.join(kaggle_path, "train_images")
    test_images_dir = os.path.join(kaggle_path, "test_images")
    train_csv_path = os.path.join(kaggle_path, "train.csv")
    
    # ============================================
    # 3. VALIDATE INPUT FILES EXIST
    # ============================================
    
    print("\n🔍 Checking input files...")
    
    # Check each required file/folder
    required_paths = [
        (train_images_dir, "train_images/ folder"),
        (train_csv_path, "train.csv file"),
    ]
    
    all_exist = True
    for path, description in required_paths:
        if os.path.exists(path):
            print(f"✅ Found {description}: {path}")
        else:
            print(f"❌ Missing {description}: {path}")
            all_exist = False
    
    if not all_exist:
        print("\n🚨 ERROR: Missing required files!")
        print("Make sure your folder structure looks like this:")
        print("your_project/")
        print("├── data/")
        print("│   ├── train_images/  ← Should contain .jpg files")
        print("│   ├── test_images/")
        print("│   └── train.csv")
        print("└── this_script.py")
        return False
    
    # Count images in train_images
    try:
        image_files = [f for f in os.listdir(train_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"📸 Found {len(image_files)} images in train_images/")
    except Exception as e:
        print(f"⚠️  Could not count images in train_images/: {e}")
        image_files = []
    
    # ============================================
    # 4. CREATE OUTPUT DIRECTORIES
    # ============================================
    
    directories = [
        os.path.join(output_path, "train", "OK"),
        os.path.join(output_path, "train", "DEFECT"),
        os.path.join(output_path, "val", "OK"),
        os.path.join(output_path, "val", "DEFECT"),
        os.path.join(output_path, "test", "OK"),
        os.path.join(output_path, "test", "DEFECT"),
    ]
    
    print(f"\n📁 Creating output directories in: {output_path}")
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
        print(f"  Created: {os.path.relpath(dir_path, current_dir)}")
    
    # ============================================
    # 5. READ AND PROCESS CSV FILE
    # ============================================
    
    print("\n📊 Reading train.csv file...")
    try:
        df = pd.read_csv(train_csv_path)
        print(f"✅ Successfully read {len(df)} rows from train.csv")
        print(f"📋 CSV columns: {list(df.columns)}")
    except Exception as e:
        print(f"❌ Failed to read train.csv: {e}")
        return False
    
    # ============================================
    # 6. IDENTIFY IMAGES WITH DEFECTS
    # ============================================
    
    print("\n🔍 Analyzing which images have defects...")
    
    # Dictionary to track defect status
    images_defects = {}
    
    # Group by ImageId
    grouped = df.groupby('ImageId')
    
    for image_id, group in grouped:
        has_defect = False
        for _, row in group.iterrows():
            encoded_pixels = row['EncodedPixels']
            if pd.notna(encoded_pixels) and str(encoded_pixels).strip() != "":
                has_defect = True
                break
        images_defects[image_id] = has_defect
    
    total_images = len(images_defects)
    defect_images = sum(images_defects.values())
    ok_images = total_images - defect_images
    
    print(f"📊 Analysis Results:")
    print(f"  • Total unique images: {total_images}")
    print(f"  • Images WITH defects: {defect_images}")
    print(f"  • Images WITHOUT defects: {ok_images}")
    print(f"  • Defect ratio: {defect_images/total_images*100:.1f}%")
    
    # ============================================
    # 7. SPLIT DATA INTO TRAIN/VAL/TEST SETS
    # ============================================
    
    print("\n🎯 Splitting data (80% train, 10% val, 10% test)...")
    
    all_images = list(images_defects.keys())
    
    # Set random seed for reproducibility
    np.random.seed(42)
    np.random.shuffle(all_images)
    
    # Calculate split indices
    train_split = int(0.8 * len(all_images))
    val_split = int(0.9 * len(all_images))  # 80% train, 10% val, 10% test
    
    train_images = all_images[:train_split]
    val_images = all_images[train_split:val_split]
    test_images = all_images[val_split:]
    
    print(f"✅ Split complete:")
    print(f"  • Training set: {len(train_images)} images")
    print(f"  • Validation set: {len(val_images)} images")
    print(f"  • Test set: {len(test_images)} images")
    
    # ============================================
    # 8. COPY IMAGES TO ORGANIZED FOLDERS
    # ============================================
    
    def copy_images(image_list, split_name):
        """
        Copy images to appropriate folders based on defect status
        """
        copied = 0
        skipped = 0
        defect_count = 0
        ok_count = 0
        
        for img_id in image_list:
            has_defect = images_defects[img_id]
            src_path = os.path.join(train_images_dir, img_id)
            
            if has_defect:
                dst_folder = "DEFECT"
                defect_count += 1
            else:
                dst_folder = "OK"
                ok_count += 1
            
            dst_path = os.path.join(output_path, split_name, dst_folder, img_id)
            
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
                copied += 1
            else:
                # Try with different extensions if .jpg doesn't work
                found = False
                for ext in ['.jpg', '.jpeg', '.png']:
                    alt_path = src_path.rsplit('.', 1)[0] + ext
                    if os.path.exists(alt_path):
                        shutil.copy2(alt_path, dst_path)
                        copied += 1
                        found = True
                        break
                
                if not found:
                    print(f"⚠️  Warning: Could not find image {img_id}")
                    skipped += 1
        
        return copied, skipped, defect_count, ok_count
    
    # Copy training images
    print("\n📤 Copying training images...")
    train_copied, train_skipped, train_defects, train_ok = copy_images(train_images, "train")
    print(f"✅ Training set:")
    print(f"   • Copied: {train_copied} images")
    print(f"   • Skipped: {train_skipped} images")
    print(f"   • OK images: {train_ok}")
    print(f"   • DEFECT images: {train_defects}")
    
    # Copy validation images
    print("\n📤 Copying validation images...")
    val_copied, val_skipped, val_defects, val_ok = copy_images(val_images, "val")
    print(f"✅ Validation set:")
    print(f"   • Copied: {val_copied} images")
    print(f"   • Skipped: {val_skipped} images")
    print(f"   • OK images: {val_ok}")
    print(f"   • DEFECT images: {val_defects}")
    
    # Copy test images
    print("\n📤 Copying test images...")
    test_copied, test_skipped, test_defects, test_ok = copy_images(test_images, "test")
    print(f"✅ Test set:")
    print(f"   • Copied: {test_copied} images")
    print(f"   • Skipped: {test_skipped} images")
    print(f"   • OK images: {test_ok}")
    print(f"   • DEFECT images: {test_defects}")
    
    # ============================================
    # 9. CREATE SUMMARY FILES
    # ============================================
    
    # Create text summary
    summary_path = os.path.join(output_path, "dataset_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("KAGGLE STEEL DEFECT DATASET - ORGANIZED STRUCTURE\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("OVERALL STATISTICS:\n")
        f.write(f"{'Total unique images:':<25} {total_images}\n")
        f.write(f"{'Images with defects:':<25} {defect_images}\n")
        f.write(f"{'Images without defects:':<25} {ok_images}\n")
        f.write(f"{'Defect percentage:':<25} {defect_images/total_images*100:.1f}%\n\n")
        
        f.write("TRAINING SET:\n")
        f.write(f"{'Total images:':<20} {len(train_images)}\n")
        f.write(f"{'OK images:':<20} {train_ok}\n")
        f.write(f"{'DEFECT images:':<20} {train_defects}\n")
        f.write(f"{'Defect percentage:':<20} {train_defects/len(train_images)*100:.1f}%\n\n")
        
        f.write("VALIDATION SET:\n")
        f.write(f"{'Total images:':<20} {len(val_images)}\n")
        f.write(f"{'OK images:':<20} {val_ok}\n")
        f.write(f"{'DEFECT images:':<20} {val_defects}\n")
        f.write(f"{'Defect percentage:':<20} {val_defects/len(val_images)*100:.1f}%\n\n")
        
        f.write("TEST SET:\n")
        f.write(f"{'Total images:':<20} {len(test_images)}\n")
        f.write(f"{'OK images:':<20} {test_ok}\n")
        f.write(f"{'DEFECT images:':<20} {test_defects}\n")
        f.write(f"{'Defect percentage:':<20} {test_defects/len(test_images)*100:.1f}%\n\n")
        
        f.write("COPYING RESULTS:\n")
        f.write(f"{'Total copied:':<20} {train_copied + val_copied + test_copied}\n")
        f.write(f"{'Total skipped:':<20} {train_skipped + val_skipped + test_skipped}\n")
        f.write(f"{'Success rate:':<20} {(train_copied + val_copied + test_copied) / total_images * 100:.1f}%\n")
    
    print(f"\n📄 Created summary file: {summary_path}")
    
    # Create YAML config for C++ project
    create_dataset_yaml(output_path, {
        'total_images': total_images,
        'defect_images': defect_images,
        'ok_images': ok_images,
        'train_ok': train_ok,
        'train_defect': train_defects,
        'val_ok': val_ok,
        'val_defect': val_defects,
        'test_ok': test_ok,
        'test_defect': test_defects
    })
    
    # ============================================
    # 10. DISPLAY FINAL RESULTS
    # ============================================
    
    print("\n" + "="*60)
    print("🎉 DATASET ORGANIZATION COMPLETE!")
    print("="*60)
    
    print("\n📊 FINAL DATASET STATISTICS:")
    print("-" * 45)
    print(f"{'Set':<12} {'Total':<8} {'OK':<8} {'DEFECT':<8} {'% Defect':<10}")
    print("-" * 45)
    
    sets = [
        ("Train", len(train_images), train_ok, train_defects),
        ("Validation", len(val_images), val_ok, val_defects),
        ("Test", len(test_images), test_ok, test_defects),
        ("TOTAL", total_images, ok_images, defect_images)
    ]
    
    for name, total, ok, defect in sets:
        defect_pct = (defect / total * 100) if total > 0 else 0
        print(f"{name:<12} {total:<8} {ok:<8} {defect:<8} {defect_pct:<8.1f}%")
    
    print("\n📁 ORGANIZED FOLDER STRUCTURE:")
    print(f"{output_path}/")
    print("├── train/")
    print("│   ├── OK/       (contains {train_ok} images)".format(train_ok=train_ok))
    print("│   └── DEFECT/   (contains {train_defects} images)".format(train_defects=train_defects))
    print("├── val/")
    print("│   ├── OK/       (contains {val_ok} images)".format(val_ok=val_ok))
    print("│   └── DEFECT/   (contains {val_defects} images)".format(val_defects=val_defects))
    print("├── test/")
    print("│   ├── OK/       (contains {test_ok} images)".format(test_ok=test_ok))
    print("│   └── DEFECT/   (contains {test_defects} images)".format(test_defects=test_defects))
    print("├── dataset_summary.txt")
    print("└── dataset_config.yaml")
    
    print("\n✅ Ready for CNN training!")
    return True

def create_dataset_yaml(output_path, stats):
    """Create YAML configuration file for C++ project"""
    
    config = {
        'dataset': {
            'name': 'Kaggle Steel Defect Detection',
            'source': 'Kaggle - Severstal Steel Defect Detection',
            'date_organized': pd.Timestamp.now().strftime('%Y-%m-%d'),
            'classes': ['OK', 'DEFECT'],
            'image_extensions': ['.jpg']
        },
        'statistics': stats,
        'paths': {
            'base': './organized',
            'train': {
                'ok': './train/OK',
                'defect': './train/DEFECT'
            },
            'val': {
                'ok': './val/OK',
                'defect': './val/DEFECT'
            },
            'test': {
                'ok': './test/OK',
                'defect': './test/DEFECT'
            }
        },
        'split_ratios': {
            'train': 0.8,
            'val': 0.1,
            'test': 0.1
        },
        'notes': 'Organized for binary classification (OK vs DEFECT)'
    }
    
    yaml_path = os.path.join(output_path, "dataset_config.yaml")
    
    try:
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"📁 Created YAML config: {yaml_path}")
    except Exception as e:
        print(f"⚠️  Could not create YAML file: {e}")
    
    return yaml_path

def main():
    """Main function to run the organization script"""
    
    print("\n" + "="*60)
    print("KAGGLE STEEL DEFECT DATASET ORGANIZER")
    print("="*60)
    print("\nThis script will organize the Kaggle dataset into the structure")
    print("needed for your CNN defect detection project.")
    print("\nLooking for dataset in: ./data/")
    print("Output will be in: ./data/organized/")
    print("\n" + "="*60)
    
    # Ask for confirmation
    response = input("\nContinue? (y/n): ").strip().lower()
    if response != 'y':
        print("Operation cancelled.")
        return
    
    # Run the organization
    success = organize_kaggle_steel_data()
    
    if success:
        print("\n" + "="*60)
        print("✅ SUCCESS! Your dataset is ready.")
        print("="*60)
        print("\nNext steps:")
        print("1. Update your config.yaml with these paths:")
        print("   paths:")
        print("     train_images: \"./data/organized/train\"")
        print("     val_images: \"./data/organized/val\"")
        print("     test_images: \"./data/organized/test\"")
        print("\n2. Start training your CNN model!")
    else:
        print("\n" + "="*60)
        print("❌ ORGANIZATION FAILED")
        print("="*60)
        print("\nPlease check:")
        print("1. Is the Kaggle dataset extracted in ./data/ ?")
        print("2. Does ./data/train_images/ contain .jpg files?")
        print("3. Is train.csv in ./data/ ?")

if __name__ == "__main__":
    # Install required packages if missing
    try:
        import pandas as pd
    except ImportError:
        print("Installing pandas...")
        import subprocess
        subprocess.check_call(["pip", "install", "pandas"])
        
    try:
        import yaml
    except ImportError:
        print("Installing pyyaml...")
        import subprocess
        subprocess.check_call(["pip", "install", "pyyaml"])
    
    # Run the main function
    main()