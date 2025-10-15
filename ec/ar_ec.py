import numpy as np
import cv2

# Import necessary functions
from loadVid import loadVid
from planarH import computeH_ransac, compositeH
from matchPics import matchPics

# Write script for Q4.1 - Augmented Reality Application

# Load the AR source video (panda/animation)
ar_source_frames = loadVid('../data/ar_source.mov')
ar_frame_count = ar_source_frames.shape[0]
print(f"AR source: {ar_frame_count} frames")

# Load the book video
book_frames = loadVid('../data/book.mov')
book_frame_count, book_H, book_W, _ = book_frames.shape
print(f"Book video: {book_frame_count} frames, size {book_W}x{book_H}")

# Load the book cover template for matching
book_cover = cv2.imread('../data/cv_cover.jpg')
cover_H, cover_W, _ = book_cover.shape
print(f"Book cover template: {cover_W}x{cover_H}")

# Setup video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # XVID is more compatible than DIVX
out = cv2.VideoWriter('ar.avi', fourcc, 20.0, (book_W, book_H))

if not out.isOpened():
    print("Error: Could not open video writer!")
    exit(1)

print("\nProcessing frames...")

for i in range(book_frame_count):
    if i % 20 == 0:
        print(f'Processing frame {i}/{book_frame_count}...')

    # Get current frames (loop AR source if needed)
    ar_frame = ar_source_frames[i % ar_frame_count]
    book_frame = book_frames[i]
    book_frame = np.squeeze(book_frame)

    ar_H, ar_W, _ = ar_frame.shape
    
    # Calculate aspect ratios
    cover_aspect = cover_W / cover_H  # Book cover aspect ratio
    ar_aspect = ar_W / ar_H            # AR source aspect ratio
    
    # Crop AR frame from center to match aspect ratio
    if ar_aspect > cover_aspect:
        new_W = int(ar_H * cover_aspect)
        start_x = (ar_W - new_W) // 2
        ar_cropped = ar_frame[:, start_x:start_x + new_W, :]
    else:
        new_H = int(ar_W / cover_aspect)
        start_y = (ar_H - new_H) // 2
        ar_cropped = ar_frame[start_y:start_y + new_H, :, :]
    
    # Resize to match book cover dimensions
    ar_resized = cv2.resize(ar_cropped, (cover_W, cover_H))

    try:
        matches, locs1, locs2 = matchPics(book_cover, book_frame)
        
        if len(matches) < 4:
            print(f"  Warning: Frame {i} has only {len(matches)} matches, skipping")
            out.write(book_frame)
            continue
        
        H2to1, inliers = computeH_ransac(
            locs1[matches[:, 0]],  # Points in book frame
            locs2[matches[:, 1]]   # Points in book cover
        )
        
        if H2to1 is None:
            print(f"  Warning: Frame {i} failed homography computation")
            out.write(book_frame)
            continue
        
        composite_img = compositeH(H2to1, ar_resized, book_frame)
        
        # Write frame to output video
        out.write(composite_img)
        
    except Exception as e:
        print(f"  Error processing frame {i}: {e}")
        out.write(book_frame)
        continue

# Release video writer
out.release()