import numpy as np
import cv2 as cv
import gradio as gr
import matplotlib.pyplot as plt

def match_features(img1, img2):
    # Convert Gradio image inputs (PIL) to OpenCV format (numpy array)
    img1 = np.array(img1.convert("L"))
    img2 = np.array(img2.convert("L"))
    
    # Initialize variables to store match counts
    orb_matches_count = 0
    sift_bf_matches_count = 0
    sift_flann_matches_count = 0

    # ORB feature matching
    orb = cv.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    # Count good ORB matches (using top 10% of matches)
    orb_matches_count = len(matches[:int(len(matches) * 0.1)])
    img3_orb = cv.drawMatches(img1, kp1, img2, kp2, matches[:orb_matches_count], None, 
                             flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # SIFT with BFMatcher
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
    sift_bf_matches_count = len(good)
    img3_sift_bf = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None, 
                                    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # SIFT with FLANN
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    matchesMask = [[0, 0] for i in range(len(matches))]
    good_matches = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]
            good_matches.append(m)
    sift_flann_matches_count = len(good_matches)
    draw_params = dict(matchColor=(0, 255, 0),
                      singlePointColor=(255, 0, 0),
                      matchesMask=matchesMask,
                      flags=cv.DrawMatchesFlags_DEFAULT)
    img3_sift_flann = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)

    # Convert images to RGB format for display in Gradio
    img3_orb = cv.cvtColor(img3_orb, cv.COLOR_BGR2RGB)
    img3_sift_bf = cv.cvtColor(img3_sift_bf, cv.COLOR_BGR2RGB)
    img3_sift_flann = cv.cvtColor(img3_sift_flann, cv.COLOR_BGR2RGB)

    # Add text with match counts to the images
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)  # White color
    thickness = 2
    
    # Add count text to each image
    def add_count_text(img, count, method_name):
        h, w = img.shape[:2]
        text = f"{method_name} Matches: {count}"
        # Get text size
        (text_width, text_height), _ = cv.getTextSize(text, font, font_scale, thickness)
        # Position text at bottom center
        x = (w - text_width) // 2
        y = h - 20  # 20 pixels from bottom
        # Add black background for better visibility
        cv.rectangle(img, (x-5, y-text_height-5), (x+text_width+5, y+5), (0,0,0), -1)
        # Add text
        cv.putText(img, text, (x, y), font, font_scale, font_color, thickness)
        return img

    img3_orb = add_count_text(img3_orb, orb_matches_count, "ORB")
    img3_sift_bf = add_count_text(img3_sift_bf, sift_bf_matches_count, "SIFT-BF")
    img3_sift_flann = add_count_text(img3_sift_flann, sift_flann_matches_count, "SIFT-FLANN")

    return img3_orb, img3_sift_bf, img3_sift_flann

# Define example image pairs with proper descriptions
examples = [
    ["charminar_1.png", "charminar_2.png"],  # Charminar example
    ["Taj_mahal_1.jpeg", "pexels-photo-28257145.webp"],  # Taj Mahal example
    ["pexels-photo-18374134.jpeg", "free-photo-of-teenage-boy-balancing-on-railway-tracks.jpeg"],  # Railway tracks example
    ["image_2A.jpg", "image_2B.jpg"]  # Generic image pair example
]

# Gradio interface with examples
iface = gr.Interface(
    fn=match_features,
    inputs=[
        gr.Image(type="pil", label="Image 1"),
        gr.Image(type="pil", label="Image 2")
    ],
    outputs=[
        gr.Image(label="ORB Matches"),
        gr.Image(label="SIFT (BFMatcher) Matches"),
        gr.Image(label="SIFT (FLANN) Matches")
    ],
    title="Image Feature Matching App",
    description="""
    Upload two images of the same subject taken from different angles to find best Feature Matches using below three different algorithms: 
    1. ORB (Oriented FAST and Rotated BRIEF)
    2. SIFT (Scale-Invariant Feature Transform) with BFMatcher
    3. SIFT with FLANN (Fast Library for Approximate Nearest Neighbors)
    """,
    examples=examples,
    cache_examples=True  # Cache examples for faster loading
)

iface.launch()