import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

def convert_dataset(images_dir, annotations_dir, output_dir, debug=False):
    """
    Convert a dataset with separate images and XML annotations directories into a folder structure
    with class-specific directories.
    
    Args:
        images_dir (str): Path to the directory containing the images
        annotations_dir (str): Path to the directory containing the XML annotations
        output_dir (str): Path to the output directory where the new structure will be created
        debug (bool): Whether to print detailed debug information
    """
    # Verify directories exist
    if not os.path.isdir(images_dir):
        print(f"Error: Images directory does not exist: {images_dir}")
        return None
    
    if not os.path.isdir(annotations_dir):
        print(f"Error: Annotations directory does not exist: {annotations_dir}")
        return None
    
    # Create the output directory structure
    os.makedirs(output_dir, exist_ok=True)
    with_mask_dir = os.path.join(output_dir, "with_mask")
    without_mask_dir = os.path.join(output_dir, "without_mask")
    incorrect_mask_dir = os.path.join(output_dir, "mask_weared_incorrect")
    
    # Create the class directories
    os.makedirs(with_mask_dir, exist_ok=True)
    os.makedirs(without_mask_dir, exist_ok=True)
    os.makedirs(incorrect_mask_dir, exist_ok=True)
    
    # Counters for statistics
    stats = {
        "total_annotations": 0,
        "processed_images": 0,
        "skipped_images": 0,
        "with_mask": 0,
        "without_mask": 0,
        "incorrect_mask": 0
    }
    
    # Get list of annotation files
    annotation_files = [f for f in os.listdir(annotations_dir) if f.lower().endswith('.xml')]
    stats["total_annotations"] = len(annotation_files)
    
    if debug:
        print(f"Found {len(annotation_files)} XML annotation files")
    
    if len(annotation_files) == 0:
        print("Error: No XML annotation files found in the annotations directory.")
        print(f"Files in directory: {os.listdir(annotations_dir)[:10]}")
        return stats
    
    # Process each annotation file
    for annotation_file in annotation_files:
        if debug:
            print(f"\nProcessing annotation: {annotation_file}")
        
        # Load the XML annotation
        annotation_path = os.path.join(annotations_dir, annotation_file)
        try:
            tree = ET.parse(annotation_path)
            root = tree.getroot()
            
            if debug:
                print(f"XML root tag: {root.tag}")
        except Exception as e:
            print(f"Error reading XML annotation file {annotation_file}: {str(e)}")
            stats["skipped_images"] += 1
            continue
        
        # Extract image filename from the XML
        image_filename = None
        
        # Common XML structures for object detection annotations
        # Option 1: Pascal VOC format
        filename_elements = root.findall('./filename') or root.findall('./path')
        if filename_elements:
            image_filename = filename_elements[0].text
            if os.path.sep in image_filename:  # Extract just the filename if it's a full path
                image_filename = os.path.basename(image_filename)
        
        # Option 2: Extract from annotation filename
        if image_filename is None:
            base_name = os.path.splitext(annotation_file)[0]
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
                potential_filename = base_name + ext
                if os.path.exists(os.path.join(images_dir, potential_filename)):
                    image_filename = potential_filename
                    if debug:
                        print(f"Found image by matching annotation name: {image_filename}")
                    break
        
        if image_filename is None:
            print(f"Error: Could not determine image filename for annotation: {annotation_file}")
            stats["skipped_images"] += 1
            continue
        
        # Determine the class label from the XML annotation
        label = None
        
        # Look for object elements with class/name information
        object_elements = root.findall('.//object') or root.findall('.//item')
        if object_elements:
            # For simplicity, we'll use the first object's class
            # For masks, often the annotation describes the mask status of the entire image
            for obj in object_elements:
                # Try different possible element names for the class
                class_elements = (obj.findall('./name') or 
                                 obj.findall('./class') or 
                                 obj.findall('./label') or
                                 obj.findall('./category'))
                
                if class_elements and class_elements[0].text:
                    label = class_elements[0].text.lower().strip()
                    if debug:
                        print(f"Found label in XML: {label}")
                    break
        
        # If no objects found, try looking directly for label elements
        if label is None:
            label_elements = (root.findall('./label') or 
                             root.findall('./class') or 
                             root.findall('./category') or
                             root.findall('./status'))
            
            if label_elements and label_elements[0].text:
                label = label_elements[0].text.lower().strip()
                if debug:
                    print(f"Found top-level label in XML: {label}")
        
        # Map the label to the three target classes
        target_dir = None
        
        # Define mappings from possible label values to our three categories
        with_mask_labels = ['with_mask', 'mask', 'face_with_mask', 'masked', 'proper', 'correct', 'yes', '1', 'true']
        without_mask_labels = ['without_mask', 'no_mask', 'face_no_mask', 'unmasked', 'none', 'no', '0', 'false']
        incorrect_mask_labels = ['mask_weared_incorrect', 'incorrect_mask', 'improper_mask', 'partial', 'wrong', 'improper']
        
        if label is not None:
            label = label.lower().strip()
            if any(mask_label in label for mask_label in with_mask_labels) or label in with_mask_labels:
                target_dir = with_mask_dir
                stats["with_mask"] += 1
            elif any(no_mask_label in label for no_mask_label in without_mask_labels) or label in without_mask_labels:
                target_dir = without_mask_dir
                stats["without_mask"] += 1
            elif any(incorrect_label in label for incorrect_label in incorrect_mask_labels) or label in incorrect_mask_labels:
                target_dir = incorrect_mask_dir
                stats["incorrect_mask"] += 1
        
        # If we couldn't determine the target directory, use a special approach for mask datasets
        if target_dir is None:
            # For typical face mask detection datasets, check if there are any mask annotations
            # in the file to determine the category
            all_text = ET.tostring(root, encoding='unicode')
            if 'mask' in all_text.lower():
                if 'incorrect' in all_text.lower() or 'improper' in all_text.lower() or 'wrong' in all_text.lower():
                    target_dir = incorrect_mask_dir
                    stats["incorrect_mask"] += 1
                elif 'without' in all_text.lower() or 'no mask' in all_text.lower():
                    target_dir = without_mask_dir
                    stats["without_mask"] += 1
                else:
                    target_dir = with_mask_dir
                    stats["with_mask"] += 1
            else:
                # If no mask-related text found, assume "without_mask"
                target_dir = without_mask_dir
                stats["without_mask"] += 1
                
            if debug and target_dir:
                print(f"Determined label by whole-file analysis: {os.path.basename(target_dir)}")
        
        # Ensure image exists
        src_path = os.path.join(images_dir, image_filename)
        if not os.path.exists(src_path):
            print(f"Error: Image not found: {src_path}")
            stats["skipped_images"] += 1
            continue
            
        # Copy the image to the appropriate class directory
        dst_path = os.path.join(target_dir, image_filename)
        
        try:
            shutil.copy2(src_path, dst_path)
            print(f"Copied {image_filename} to {os.path.relpath(target_dir, output_dir)}")
            stats["processed_images"] += 1
        except Exception as e:
            print(f"Error copying {src_path} to {dst_path}: {str(e)}")
            stats["skipped_images"] += 1
    
    # Print statistics
    print("\n--- Conversion Statistics ---")
    print(f"Total annotation files: {stats['total_annotations']}")
    print(f"Successfully processed images: {stats['processed_images']}")
    print(f"Skipped images: {stats['skipped_images']}")
    print(f"Images by category:")
    print(f"  - with_mask: {stats['with_mask']}")
    print(f"  - without_mask: {stats['without_mask']}")
    print(f"  - mask_weared_incorrect: {stats['incorrect_mask']}")
    
    return stats

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert mask dataset to class directory structure')
    parser.add_argument('--images', required=True, help='Path to the images directory')
    parser.add_argument('--annotations', required=True, help='Path to the annotations directory')
    parser.add_argument('--output', required=True, help='Path to the output directory')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    
    args = parser.parse_args()
    
    print(f"Converting dataset from:")
    print(f"  Images: {args.images}")
    print(f"  Annotations: {args.annotations}")
    print(f"  To: {args.output}")
    
    stats = convert_dataset(args.images, args.annotations, args.output, args.debug)
    
    print("\nDataset conversion complete!")