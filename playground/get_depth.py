# Step 1: Modify Dataset class, add preprocessing to fixed size
class CustomImageDataset(Dataset):
    """
    Dataset with preprocessing to fixed size (336x322)
    """
    # Depth anything model needs input to be divisible by 14
    # Aspect ratio of InsPLAD in average is 1.05
    
    def __init__(self, image_files, target_size=(336, 322)):
        self.image_files = image_files
        self.target_width, self.target_height = target_size

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Warning: Could not load {img_path}, returning None.")
                return None
            
            # Record original dimensions (for later restoration)
            original_shape = img.shape[:2]  # (H, W)
            
            # Convert to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Preprocess to fixed size: use zero padding to maintain aspect ratio
            img_resized = self.resize_with_padding(img, self.target_width, self.target_height)
            
            # Convert to tensor
            img_tensor = torch.from_numpy(img_resized / 255.0).permute(2, 0, 1).float()
            
            return img_tensor, original_shape, img_path
            
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return None
    
    def resize_with_padding(self, img, target_w, target_h):
        """
        Resize image to target size while maintaining aspect ratio, pad insufficient areas with black
        """
        h, w = img.shape[:2]
        
        # Calculate scaling ratio to maintain aspect ratio
        scale = min(target_w / w, target_h / h)
        
        # Calculate new dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Scale the image
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create black background with target size
        result = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
        # Calculate center position
        start_x = (target_w - new_w) // 2
        start_y = (target_h - new_h) // 2
        
        # Place scaled image at center
        result[start_y:start_y + new_h, start_x:start_x + new_w] = resized
        
        return result

# Step 2: Simplify collate_fn (since all images are now the same size)
def simple_collate_fn(batch):
    """
    Simple batching function: all images are the same size, can be stacked directly
    """
    # Filter out failed loading samples
    batch = [item for item in batch if item is not None]
    if not batch:
        return None, None, None

    img_tensors, original_shapes, img_paths = zip(*batch)
    
    # All images are the same size, stack directly
    batched_tensor = torch.stack(img_tensors, dim=0)
        
    return batched_tensor, original_shapes, img_paths

# Step 3: Modify processing function
def process_dataset_fixed_size(input_dir, output_dir, target_size=(336, 322), batch_size=8, num_workers=4):
    """
    Efficient batch processing using fixed size preprocessing
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    image_files = [p for ext in img_extensions for p in input_path.rglob(f'*{ext}')]
    print(f"Found {len(image_files)} images to process.")
    print(f"Target size: {target_size[0]} x {target_size[1]}")

    # Create Dataset and DataLoader
    dataset = CustomImageDataset(image_files, target_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=simple_collate_fn,
        pin_memory=True if device.type == 'cuda' else False
    )

    processed_count = 0
    for batched_tensor, original_shapes, img_paths in tqdm(dataloader, desc="Processing batches"):
        if batched_tensor is None:
            continue

        # Move data to GPU
        batched_tensor = batched_tensor.to(device)

        with torch.no_grad():
            depth_preds = model(batched_tensor)

        # Save results
        for j in range(len(img_paths)):
            try:
                img_path = Path(img_paths[j])
                original_h, original_w = original_shapes[j]
                
                # Get single prediction result back from GPU
                depth_pred = depth_preds[j].cpu().numpy()
                
                # Restore depth map to original image size
                depth_original_size = restore_original_size(
                    depth_pred, original_w, original_h, target_size
                )

                # Save file (same logic as before)
                relative_path = img_path.relative_to(input_path)
                output_img_dir = output_path / relative_path.parent
                output_img_dir.mkdir(parents=True, exist_ok=True)
                output_file_base = output_img_dir / img_path.stem
                
                depth_normalized_uint8 = cv2.normalize(depth_original_size, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

                # Save uint8 format npy file
                np.save(f"{output_file_base}_depth.npy", depth_normalized_uint8)
                
                # Generate colored visualization image using normalized depth map
                depth_colored = cv2.applyColorMap(depth_normalized_uint8, cv2.COLORMAP_PLASMA)
                cv2.imwrite(f"{output_file_base}_depth.png", depth_colored)
                
                processed_count += 1
                
            except Exception as e:
                print(f"Error saving depth for {img_paths[j]}: {e}")
                continue

    print(f"Successfully processed {processed_count} images.")
    return processed_count

def restore_original_size(depth_pred, original_w, original_h, target_size):
    """
    Restore depth map from fixed size back to original image size
    """
    target_w, target_h = target_size
    
    # Calculate scaling and position of original image within target size
    scale = min(target_w / original_w, target_h / original_h)
    scaled_w = int(original_w * scale)
    scaled_h = int(original_h * scale)
    
    # Calculate position in target image
    start_x = (target_w - scaled_w) // 2
    start_y = (target_h - scaled_h) // 2
    
    # Extract valid region from depth map
    depth_cropped = depth_pred[start_y:start_y + scaled_h, start_x:start_x + scaled_w]
    
    # Scale back to original size
    depth_original = cv2.resize(depth_cropped, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
    
    return depth_original
